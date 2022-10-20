import argparse
import ast
import copy
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torch import nn

import sys
from pathlib import Path
utils_dir = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
sys.path.append(str(utils_dir))
from utils import str2bool

from .loss import setup_loss
from .metric import EarthNetScore

None2nan= lambda n: np.nan if n is None else n

class STFTask(pl.LightningModule):

    def __init__(self, model: nn.Module, hparams: argparse.Namespace):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.model = model

        if hparams.pred_dir is None:
            self.pred_dir = Path(
                self.logger.log_dir) / "predictions" if self.logger is not None \
                else Path.cwd() / "experiments" / "predictions"
        else:
            self.pred_dir = Path(self.hparams.pred_dir)

        self.loss = setup_loss(hparams.loss)

        self.context_length = hparams.context_length
        self.target_length = hparams.target_length

        self.n_stochastic_preds = hparams.n_stochastic_preds

        self.current_filepaths = []

        self.compute_ens_on_test = hparams.compute_ens_on_test
        self.ens = EarthNetScore(use_multiprocessing=hparams.use_multiprocessing)

    @staticmethod
    def add_task_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument('--pred_dir', type=str, default=None)

        parser.add_argument('--loss', type=ast.literal_eval,
                            default='{"name": "masked", "args": {"distance_type": "L1"}}')

        parser.add_argument('--context_length', type=int, default=10)
        parser.add_argument('--target_length', type=int, default=20)
        parser.add_argument('--n_stochastic_preds', type=int, default=10)

        parser.add_argument('--n_log_batches', type=int, default=2)
        parser.add_argument('--use_multiprocessing', type = bool, default = False)
        parser.add_argument('--time_downsample', type = int, default = 1)

        parser.add_argument(
            '--optimization',
            type=ast.literal_eval,
            default='{"optimizer": [{"name": "Adam", "args:" {"lr": 0.0001, "betas": (0.9, 0.999)} }], '
                    '"lr_shedule": [{"name": "multistep", "args": {"milestones": [25, 40], "gamma": 0.1} }]}')

        parser.add_argument('--compute_ens_on_test', type=str2bool, default=False)

        return parser

    def forward(self, data, pred_start: int = 0, n_preds: Optional[int] = None):
        return self.model(data, pred_start=pred_start, n_preds=n_preds, crt_epoch=self.current_epoch)

    def configure_optimizers(self):
        optimizers = [getattr(torch.optim, o["name"])(self.parameters(), **o["args"]) for o in
                      self.hparams.optimization["optimizer"]]
        shedulers = [getattr(torch.optim.lr_scheduler, s["name"])(optimizers[i], **s["args"]) for i, s in
                     enumerate(self.hparams.optimization["lr_shedule"])]
        return optimizers, shedulers

    def training_step(self, batch, batch_idx):
        preds, aux = self(batch, n_preds=self.context_length + self.target_length)

        loss, logs = self.loss(preds, batch, aux, current_step=self.global_step)
        self.log_dict({l + "_train_step": logs[l] for l in logs}, on_epoch=False, on_step=True, prog_bar=True,
                      batch_size=len(["dynamic"][0]))

        logs_epoch = {l + "_train_epoch": logs[l] for l in logs}
        logs_epoch['step'] = float(self.current_epoch)
        self.log_dict(logs_epoch, on_epoch=True, on_step=False, batch_size=len(["dynamic"][0]))

        return loss

    # def on_epoch_start(self):
    #     # https://github.com/PyTorchLightning/pytorch-lightning/issues/2189
    #     print('\n')

    def validation_step(self, batch, batch_idx):
        data = copy.deepcopy(batch)

        data["dynamic"][0] = data["dynamic"][0][:, :self.context_length, ...]
        data["dynamic_mask"][0] = data["dynamic_mask"][0][:, :self.context_length, ...]

        all_logs = []
        log_viz = self.trainer.is_global_zero and (batch_idx < self.hparams.n_log_batches)
        all_viz = []
        for i in range(self.n_stochastic_preds):
            preds, aux = self(data, pred_start=self.context_length, n_preds=self.target_length)
            all_logs.append(self.loss(preds, batch, aux)[1])
            if log_viz:
                self.ens.compute_on_step = True
                scores = self.ens(preds, batch)
                self.ens.compute_on_step = False
                all_viz.append((preds, scores))
            else:
                self.ens(preds, batch)

        mean_logs = {l: torch.tensor([log[l] for log in all_logs], dtype=torch.float32, device=self.device).mean()
                     for l in all_logs[0]}
        mean_logs_epoch = {l + "_val_epoch": mean_logs[l] for l in mean_logs}
        mean_logs_epoch['step'] = float(self.current_epoch)
        self.log_dict(mean_logs_epoch, on_epoch=True, on_step=False,
                      batch_size=len(data["dynamic"][0]))  # TODO check sync_dist=True

        if log_viz:
            self.log_viz(all_viz, batch, batch_idx)

    def validation_epoch_end(self, validation_step_outputs):
        ens_scores = self.ens.compute()
        ens_scores['step'] = float(self.current_epoch)
        self.log_dict(ens_scores, on_step=False, on_epoch=True)
        self.ens.reset()

    def test_epoch_end(self, outputs):
        if self.compute_ens_on_test: 
            scores= self.ens.compute()
            print('Test scores:', scores)
            with open(f'{str(self.pred_dir)}/scores.txt', 'a') as f:
                f.write(str(scores))
            self.ens.reset()

    def test_step(self, batch, batch_idx):
        self.ens.compute_on_step = False
        for i in range(self.n_stochastic_preds):
            preds, _ = self(batch, pred_start=self.context_length, n_preds=self.target_length)
            if self.compute_ens_on_test: self.ens(preds, batch)
            for j, pred in enumerate(preds):
                cubename = batch["cubename"][j]
                cube_dir = self.pred_dir / 'target' / cubename[:5]
                cube_dir.mkdir(parents=True, exist_ok=True)
                cube_path = cube_dir / f"pred{i + 1}_target_{cubename.replace('.zip', '')}"
                pred= pred.cpu().numpy()
                if self.hparams.time_downsample > 1:
                    t, c, h, w= pred.shape
                    pred2= np.zeros((self.hparams.target_length*self.hparams.time_downsample, c, h, w))
                    if False: #self.hparams.time_downsample == 2: #Custom processing for n=2
                        #Assuming we used averaging for downsampling, and not just holding
                        #TO DO
                        pass
                    else:
                        for k in range(self.hparams.time_downsample):
                            pred2[k::self.hparams.time_downsample]= pred
                    pred= pred2
                #save(str(cube_path), pred.permute(2,3,1,0), 'zip')
                np.savez_compressed(str(cube_path) + '.npz', highresdynamic=np.transpose(pred,(2, 3, 1, 0)))

    def log_viz(self, viz_data, batch, batch_idx):
        tensorboard_logger = self.logger.experiment
        targs = batch["dynamic"][0]
        nrow= 10
        for i, (preds, scores) in enumerate(viz_data):
            for j in range(preds.shape[0]):
                # Predictions vs GT RGB
                rgb = torch.cat([preds[j,:,2,...].unsqueeze(1)*10000,preds[j,:,1,...].unsqueeze(1)*10000,preds[j,:,0,...].unsqueeze(1)*10000],dim = 1)
                grid = torchvision.utils.make_grid(rgb, nrow = nrow, normalize = True, value_range = (0,5000))
                if i == 0:
                    rgb_gt = torch.cat([targs[j,:,2,...].unsqueeze(1)*10000,targs[j,:,1,...].unsqueeze(1)*10000,targs[j,:,0,...].unsqueeze(1)*10000],dim = 1)
                    grid_gt = torchvision.utils.make_grid(rgb_gt[-preds.shape[1]:], nrow = nrow, normalize = True, value_range = (0,5000))
                    grid = torch.cat([grid, grid_gt], dim = -2)
                    ENS, MAD, OLS, EMD, SSIM= scores[j]['ENS'], scores[j]['MAD'], scores[j]['OLS'], scores[j]['EMD'], scores[j]['SSIM']
                    ENS, MAD, OLS, EMD, SSIM= None2nan(ENS), None2nan(MAD), None2nan(OLS), None2nan(EMD), None2nan(SSIM)
                text = f"Cube: {scores[j]['name']} ENS: {ENS:.4f} MAD: {MAD:.4f} OLS: {OLS:.4f} EMD: {EMD:.4f} SSIM: {SSIM:.4f}"
                text = torch.tensor(self.text_phantom(text, width = grid.shape[-1])).type_as(grid).permute(2,0,1)
                grid = torch.cat([grid, text], dim = -2)
                tensorboard_logger.add_image(f"Cube: {batch_idx*preds.shape[0] + j} RGB Preds, Sample: {i}", grid, self.current_epoch)

                # Predictions vs GT NDVI
                ndvi = (preds[j,:,3,...] - preds[j,:,2,...])/(preds[j,:,3,...] + preds[j,:,2,...]+1e-8)
                grid = torchvision.utils.make_grid(ndvi.unsqueeze(1), nrow = nrow)
                if i == 0:
                    ndvi_gt = (targs[j,:,3,...] - targs[j,:,2,...])/(targs[j,:,3,...] + targs[j,:,2,...])
                    grid_gt2 = torchvision.utils.make_grid(ndvi_gt.unsqueeze(1)[-preds.shape[1]:], nrow = nrow)
                    grid = torch.cat([grid, grid_gt2], dim = -2)
                grid = torch.cat([grid, text], dim = -2)
                tensorboard_logger.add_image(f"Cube: {batch_idx*preds.shape[0] + j} NDVI Preds, Sample: {i}", grid, self.current_epoch)

    def text_phantom(self, text, width):
        # Create font
        pil_font = ImageFont.load_default()  #
        text_width, text_height = pil_font.getsize(text)

        # create a blank canvas with extra space between lines
        canvas = Image.new('RGB', [width, text_height], (255, 255, 255))

        # draw the text onto the canvas
        draw = ImageDraw.Draw(canvas)
        offset = ((width - text_width) // 2, 0)
        white = "#000000"
        draw.text(offset, text, font=pil_font, fill=white)

        # Convert the canvas into an array with values in [0, 1]
        return (255 - np.asarray(canvas)) / 255.0
