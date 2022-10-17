import argparse
import ast
from typing import Optional, Union

import torch
from torch import nn

import sys
from pathlib import Path
utils_dir = Path.cwd().parent.parent.parent.parent
#print(f'File: {__file__}; utils dir: {utils_dir}')
sys.path.append(str(utils_dir))
from utils import str2bool
from .ConvLSTM import ConvLSTM


class ConvLSTMen(nn.Module):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()

        self.hparams = hparams
        self.hparams.args['kernel_size'] = tuple(self.hparams.args['kernel_size'])

        # set the input dimension, depending on which inputs are used
        input_dim = 4 # RGB NIR
        if self.hparams.use_clim_vars:
            input_dim += 5 #Clim vars
        if self.hparams.use_dem_as_dynamic:
            input_dim += 1 #DEM
        if self.hparams.use_mask_as_input:
            input_dim += 1 #Good quality mask
        if self.hparams.use_scalars:
            input_dim += 4 #(long, lat, daysin, daycos)

        self.hparams.args['input_dim'] = input_dim

        # preprocess the hidden dimensions if not given as list
        hidden_dim = self.hparams.args['hidden_dim']
        if not isinstance(hidden_dim, list):
            assert isinstance(hidden_dim, int)
            self.hparams.args['hidden_dim'] = [hidden_dim for _ in range(self.hparams.args['num_layers'])]

        # add a final layer for predicting the next frame
        self.hparams.args['hidden_dim'].append(4)
        self.hparams.args['num_layers'] += 1

        print(f"Building ConvLSTM: {self.hparams.args}")
        self.conv_lstm = ConvLSTM(**self.hparams.args)
        print(self.conv_lstm)

        self.upsample = nn.Upsample(size=(128, 128))

    @staticmethod
    def add_model_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument("--name", type=str)
        parser.add_argument("--args", type=ast.literal_eval)
        parser.add_argument("--context_length", type=int, default=10)
        parser.add_argument("--target_length", type=int, default=20)
        parser.add_argument("--use_clim_vars", type=str2bool, default=True)
        parser.add_argument("--use_mask_as_input", type=str2bool, default=False)
        parser.add_argument("--use_scalars", type=str2bool, default=True)
        parser.add_argument("--use_dem_as_dynamic", type=str2bool, default=True)

        return parser

    def get_output_frame(self, hidden_state):
        # use the last hidden state
        h_last = hidden_state[-1][0]

        # ensure that the values are positive
        pred_frame = torch.relu(h_last)

        return pred_frame

    def forward(self, data, pred_start: int = 0, n_preds=None, crt_epoch=None, return_info=False):
        internal_info = {'gates': [], 'states': []}

        # prepare the data
        satimgs = data["dynamic"][0][:, :self.hparams.context_length]
        b, t, c, h, w = satimgs.shape

        dem = None
        if self.hparams.use_dem_as_dynamic or self.hparams.use_dem_static:
            dem = data["static"][0]

        clims = None
        if self.hparams.use_clim_vars:
            clims = data["dynamic"][1][:, :, :5, ...]

            _, t2, c2, h2, w2 = clims.shape
            if w2 != 2: #Compatibility with full maps
                clims = clims[...,39:41,39:41]
            clims = self.upsample(clims.reshape(b, t2*c2, 2, 2))
            clims = clims.reshape(b, t2, c2, h, w)

        #If mask is used for training, it must be all ones in prediction (i.e.: all nn-predicted)
        #data is assumed to be good!
        if self.hparams.use_mask_as_input:
            mask_og= data["dynamic_mask"][0]
            _, _, c3, h3, w3= mask_og.shape
            mask= torch.cat([mask_og[:, :self.hparams.context_length, ...],
                             torch.ones((b, self.hparams.target_length, c3, w3, h3), 
                             dtype=mask_og.dtype, device=mask_og.device)], axis=1)
        scalars= data["scalars"]

        # Step 1: encode the context frames
        hidden_state = self.conv_lstm._init_hidden(batch_size=b)
        for t in range(self.hparams.context_length):
            inputs = []

            crt_frame = satimgs[:, t]
            inputs.append(crt_frame)

            if self.hparams.use_dem_as_dynamic:
                inputs.append(dem)

            if self.hparams.use_clim_vars:
                inputs.append(clims[:, t])

            if self.hparams.use_mask_as_input:
                inputs.append(mask[:, t, [0]])
            
            if self.hparams.use_scalars:
                inputs.append(scalars) #Add scalars 

            inputs = torch.cat(inputs, dim=1)

            res = self.conv_lstm(inputs, hidden_state, return_gates=return_info)
            if return_info:
                h_last, hidden_state, gates = res
                internal_info['states'].append([(x[0].cpu(), x[1].cpu()) for x in hidden_state])
                internal_info['gates'].append(gates)
            else:
                h_last, hidden_state = res

        # Step 2: conditional forecast
        # get the output frame based on the current hidden states
        pred_frame = self.get_output_frame(hidden_state=hidden_state)

        # forecast loop, conditioned on weather
        preds = [pred_frame]
        for t in range(self.hparams.target_length - 1):
            inputs = [preds[-1]]

            if self.hparams.use_dem_as_dynamic:
                inputs.append(dem)

            if self.hparams.use_clim_vars:
                inputs.append(clims[:, self.hparams.context_length + t, ...])

            if self.hparams.use_mask_as_input:
                inputs.append(mask[:, self.hparams.context_length + t, [0], ...])

            if self.hparams.use_scalars:
                inputs.append(scalars) #Add scalars 

            inputs = torch.cat(inputs, dim=1)

            res = self.conv_lstm(inputs, hidden_state, return_gates=return_info)
            if return_info:
                h_last, hidden_state, gates = res
                internal_info['states'].append([(x[0].cpu(), x[1].cpu()) for x in hidden_state])
                internal_info['gates'].append(gates)
            else:
                h_last, hidden_state = res

            # get the next frame
            pred_frame = self.get_output_frame(hidden_state=hidden_state)
            preds.append(pred_frame)
        preds = torch.cat([frame[:, None, ...] for frame in preds], dim=1)

        if return_info:
            return preds, {}, internal_info
        return preds, {}
