import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import warnings

import numpy as np
import pytorch_lightning as pl
import yaml

from models.weather2land.code.utils.utils_general import str2bool

# go to the project root directory and add it to path
proj_root_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(proj_root_dir))
os.chdir(proj_root_dir)

calc_dir = Path.cwd().parent.parent.parent.parent / 'earthnet-toolkit'
sys.path.append(str(calc_dir))
print(f'File: {__file__}; calc_dir: {calc_dir}')
from earthnet.parallel_score import EarthNetScore

from model.conv_lstm.conv_lstm_en import ConvLSTMen
from task.data import EarthNet2021DataModule
from task.stf import STFTask

__MODELS__ = {
    'conv_lstm': ConvLSTMen,
}

__TRACKS__ = {
        "iid": "iid_test_split/",
        "ood": "ood_test_split/",
        "ex": "extreme_test_split/",
        "sea": "seasonal_test_split/",
    }


def test_model(setting_dict: dict, checkpoint: str):
    # Data
    data_args = ["--{}={}".format(key, value) for key, value in setting_dict["Data"].items()]
    data_parser = ArgumentParser()
    data_parser = EarthNet2021DataModule.add_data_specific_args(data_parser)
    data_params = data_parser.parse_args(data_args)
    dm = EarthNet2021DataModule(data_params)

    # Model
    model_args = ["--{}={}".format(key, value) for key, value in setting_dict["Model"].items()]
    model_parser = ArgumentParser()
    model_parser = __MODELS__[setting_dict["Architecture"]].add_model_specific_args(model_parser)
    model_params = model_parser.parse_args(model_args)
    vars(model_params)['context_length']= setting_dict['context_length']
    vars(model_params)['target_length']= setting_dict['target_length']
    model = __MODELS__[setting_dict["Architecture"]](model_params)

    # Task
    task_args = ["--{}={}".format(key, value) for key, value in setting_dict["Task"].items()]
    task_parser = ArgumentParser()
    task_parser = STFTask.add_task_specific_args(task_parser)
    task_params = task_parser.parse_args(task_args)
    vars(task_params)['context_length']= setting_dict['context_length']
    vars(task_params)['target_length']= setting_dict['target_length']
    vars(task_params)['time_downsample']= setting_dict['time_downsample']
    task = STFTask(model=model, hparams=task_params)
    task.load_from_checkpoint(checkpoint_path=checkpoint, context_length=setting_dict["context_length"],
                              target_length=setting_dict["target_length"], model=model, hparams=task_params)

    # Trainer
    trainer_dict = setting_dict["Trainer"]
    trainer_dict["precision"] = 16 if dm.hparams.fp16 else 32
    trainer = pl.Trainer(**trainer_dict)

    trainer.test(model=task, datamodule=dm, ckpt_path=None)


def get_best_model_ckpt(checkpoint_dir):
    ckpt_list = sorted(list(Path(checkpoint_dir).glob('*.ckpt')))
    ens_list = np.array([float(p.stem.split('EarthNetScore=')[1]) for p in ckpt_list])

    # get the index of the last maximum value
    i_max = len(ens_list) - np.argmax(ens_list[::-1]) - 1
    ckpt_best = ckpt_list[i_max]

    return ckpt_best


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--setting', type=str, metavar='path/to/setting.yaml', help='yaml with all settings')
    parser.add_argument('--track', type=str, default='all', metavar='iid|ood|ex|sea|all',
                        help='which track to test: either iid, ood, ex, sea or all for testing all of them')

    #If version is provided, the rest of the arguments are not needed
    parser.add_argument('--version', type=int, default=None, metavar='0',
                        help='Version of the experiment (int) assigned by the logger')

    #Rest of the arguments
    parser.add_argument('--checkpoint', type=str, default=None, metavar='path/to/checkpoint', help='checkpoint file')
    parser.add_argument('--checkpoint_dir', type=str, metavar='path/to/checkpoint_dir',
                        help='a directory from which the model with the best ENS score will be selected '
                             '(alternative to checkpoint_file)', default=None)
    parser.add_argument('--pred_dir', type=str, default=None, metavar='path/to/predictions/directory/',
                        help='Path where to save predictions')
    parser.add_argument('--evaluate', type=str2bool, default=False, help='Evaluate predictions too?')

    args = parser.parse_args()

    with open(args.setting, 'r') as fp:
        setting_dict = yaml.load(fp, Loader=yaml.FullLoader)

    if args.track == 'all':
        tracks= ['iid','ood','ex','sea']
    else:
        tracks= [args.track]

    for track in tracks:
        print('Testing track:', track)
        #If version is provided 
        if args.version is not None:
            if not f'version_{args.version}' in args.setting:
                warnings.warn(f'You should use the settings.yaml file contained within the version_{args.version} directory')
            setting_dict["Task"]["pred_dir"]= os.path.join(setting_dict["Logger"]["save_dir"], 
                        setting_dict["Logger"]["name"], f'version_{args.version}', 'predictions', __TRACKS__[track])

            checkpoint_dir= os.path.join(setting_dict["Logger"]["save_dir"], 
                        setting_dict["Logger"]["name"], f'version_{args.version}', 'checkpoints')
            checkpoint_file = get_best_model_ckpt(checkpoint_dir)
            print(f"Best checkpoint: {checkpoint_file}")
        else:
            if args.pred_dir is not None:
                setting_dict["Task"]["pred_dir"] = args.pred_dir
            else: 
                setting_dict["Task"]["pred_dir"]= os.path.join(setting_dict["Logger"]["save_dir"], 
                        setting_dict["Logger"]["name"], f'version_{args.version}', 'predictions', __TRACKS__[track])

            if args.checkpoint_dir is None:
                checkpoint_file = args.checkpoint
            else:
                checkpoint_file = get_best_model_ckpt(args.checkpoint_dir)
                print(f"Best checkpoint: {checkpoint_file}")
        print(f'Saving predictions at: {setting_dict["Task"]["pred_dir"]}')

        setting_dict["Data"]["test_track"] = track
        test_model(setting_dict, checkpoint_file)

        if args.evaluate:
            #Output dir
            save_dir = os.path.join(setting_dict["Task"]["pred_dir"].replace('predictions', 'scores'))
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            data_output_file = os.path.join(save_dir, 'data.json')
            ens_output_file = os.path.join(save_dir, 'model.json')

            EarthNetScore.get_ENS(setting_dict["Task"]["pred_dir"], os.path.join(setting_dict["dataset_dir"], __TRACKS__[track], 'target'), 
                                  n_workers=8, data_output_file=data_output_file, ens_output_file=ens_output_file)