import argparse
import multiprocessing
import re
from typing import Union, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import warnings

import sys
from pathlib import Path
utils_dir = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
sys.path.append(str(utils_dir))
from utils import str2bool, load

class EarthNet2021Dataset(Dataset):
    def __init__(self, folder: Union[Path, str], noisy_masked_pixels = False, 
                use_meso_static_as_dynamic = False, fp16 = False, 
                online_data_augmentation=True, online_time_downsample=1, time_downsample=1):
        if not isinstance(folder, Path):
            folder = Path(folder)
        assert (not {"target","context"}.issubset(set([d.name for d in folder.glob("*") if d.is_dir()])))
        self.filepaths = sorted(list(folder.glob("**/*.zip")))
        print(f'Found {len(self.filepaths)} samples')
        assert len(self.filepaths) > 0, f"No files found at: {str(folder)}"

        self.noisy_masked_pixels = noisy_masked_pixels
        self.use_meso_static_as_dynamic = use_meso_static_as_dynamic
        self.online_data_augmentation= online_data_augmentation
        self.online_time_downsample= online_time_downsample
        self.time_downsample= time_downsample
        self.type = np.float16 if fp16 else np.float32
        self.rng = np.random.default_rng(seed=42)

    def __getitem__(self, idx: int) -> dict:
        
        filepath = self.filepaths[idx]

        npz = load(str(filepath.resolve()))

        #(x, y, c, t) -> (t, c, x, y) 
        highresdynamic = np.transpose(npz["highresdynamic"],(3,2,0,1)).astype(self.type)[:,[2,1,0,3]] #RGB -> GBR
        highresstatic = np.transpose(npz["highresstatic"],(2,0,1)).astype(self.type)
        mesodynamic = np.transpose(npz["mesodynamic"],(3,2,0,1)).astype(self.type)
        mesostatic = np.transpose(npz["mesostatic"],(2,0,1)).astype(self.type)
        highresmask= np.transpose(npz["highresmask"],(2,0,1)).astype(self.type)

        #Good quality mask (4 channels)
        masks = ((1 - highresmask)[:,np.newaxis,:,:]).repeat(4,1)

        #RGB + NIR (4 channels)
        images = highresdynamic
        
        #Fix nans in meso
        images[np.isnan(images)]= 0
        mesodynamic[np.isnan(mesodynamic)] = 0
        highresstatic[np.isnan(highresstatic)] = 0
        mesostatic[np.isnan(mesostatic)] = 0

        #Left for backwards compatibility only
        if self.noisy_masked_pixels:            
            warnings.warn('noisy_masked_pixels is no longer needed', DeprecationWarning)
            # images = np.transpose(images,(1,0,2,3))
            # all_pixels = images[np.transpose(masks, (1,0,2,3)) == 1].reshape(4,-1)
            # all_pixels = np.stack(int(images.size/all_pixels.size+1)*[all_pixels],axis = 1)
            # all_pixels = all_pixels.reshape(4,-1)
            # all_pixels = all_pixels.transpose(1,0)
            # np.random.shuffle(all_pixels)
            # all_pixels = all_pixels.transpose(1,0)
            # all_pixels = all_pixels[:,:images.size//4].reshape(*images.shape)
            # images = np.where(np.transpose(masks, (1,0,2,3)) == 0, all_pixels, images)
            # images = np.transpose(images,(1,0,2,3))

        if self.use_meso_static_as_dynamic:
            mesodynamic = np.concatenate([mesodynamic, mesostatic[np.newaxis, :, :, :].repeat(mesodynamic.shape[0], 0)], axis = 1)

        #Data augmentation
        if self.online_data_augmentation:
            #Rotate 0-270
            rot_times= self.rng.integers(0,3)
            if rot_times > 0:
                images= np.rot90(images, k=rot_times, axes=(-1, -2))
                mesodynamic= np.rot90(mesodynamic, k=rot_times, axes=(-1, -2))
                masks= np.rot90(masks, k=rot_times, axes=(-1, -2))
                highresstatic= np.rot90(highresstatic, k=rot_times, axes=(-1, -2))
                mesostatic= np.rot90(mesostatic, k=rot_times, axes=(-1, -2))

            #Flip vertically
            if self.rng.integers(0,2): 
                images= np.flip(images, axis=-1)
                mesodynamic= np.flip(mesodynamic, axis=-1)
                masks= np.flip(masks, axis=-1)
                highresstatic= np.flip(highresstatic, axis=-1)
                mesostatic= np.flip(mesostatic, axis=-1)

        #Perform online time downsampling
        #Note that the temporary axis must be to the right of the original axis, otherwise this does not work!
        if self.online_time_downsample > 1:
            images= np.mean(images.reshape( 
                            images.shape[0]//self.online_time_downsample, self.online_time_downsample,
                            *images.shape[1:]), axis=1)
            # images= images[self.online_time_downsample-1 :: self.online_time_downsample]
            mesodynamic= np.mean(mesodynamic.reshape(
                            mesodynamic.shape[0]//self.online_time_downsample, 
                            self.online_time_downsample,
                            *mesodynamic.shape[1:]), axis=1)
            masks= (np.mean(masks.reshape(
                            masks.shape[0]//self.online_time_downsample, 
                            self.online_time_downsample, 
                            *masks.shape[1:]), axis=1) > 0.49).astype(self.type)

        #Standarization of coordinate data
        min_long, max_long, min_lat, max_lat= -15, 30, 35, 70
        coords= npz["coordinates"]
        coordinates= ((coords[0] - min_long) / (max_long - min_long),
                      (coords[1] - min_lat) / (max_lat - min_lat))

        #Cosine encoding of day of year
        _, _, w, h= highresdynamic.shape
        t2, _, _, _= mesodynamic.shape
        if 'start_date' not in npz.keys(): #backwards compatibility
            end_day= npz['date'].timetuple().tm_yday
            start_day= end_day - self.time_downsample * t2 * 5
        else:
            start_day= npz['start_date'].timetuple().tm_yday
            end_day= npz['end_date'].timetuple().tm_yday
        day_range= 2 * np.pi * np.arange(start_day, end_day-0.1, step=self.time_downsample*5) / 365.0  

        scalars= np.ones((t2, 4, w, h)).astype(self.type)
        scalars[:,0]*= coords[0]
        scalars[:,1]*= coords[1]
        scalars[:,2]*= np.sin(day_range)[:, None, None]
        scalars[:,3]*= np.cos(day_range)[:, None, None]
        scalars= scalars.astype(self.type)

        data = {
            "dynamic": [
                torch.from_numpy(images.copy()),
                torch.from_numpy(mesodynamic.copy())
            ],
            "dynamic_mask": [
                torch.from_numpy(masks.copy())
            ],
            "static": [
                torch.from_numpy(highresstatic.copy()),
                torch.from_numpy(mesostatic.copy())
            ] if not self.use_meso_static_as_dynamic else [
                torch.from_numpy(highresstatic.copy())
            ],
            "scalars": torch.from_numpy(scalars),
            "filepath": str(filepath),
            "cubename": self.__name_getter(filepath),
            "long": coordinates[0],
            "lat": coordinates[1],
            "dayofyear": start_day,
        }

        return data

    def __len__(self) -> int:
        return len(self.filepaths)

    def __name_getter(self, path: Path) -> str:
        """Helper function gets Cubename from a Path

        Args:
            path (Path): One of Path/to/cubename.npz and Path/to/experiment_cubename.npz

        Returns:
            [str]: cubename (has format tile_startyear_startmonth_startday_endyear_endmonth_endday_hrxmin_hrxmax_hrymin_hrymax_mesoxmin_mesoxmax_mesoymin_mesoymax.npz)
        """
        components = path.name.split("_")
        regex = re.compile('\d{2}[A-Z]{3}')
        if bool(regex.match(components[0])):
            return path.name
        else:
            assert (bool(regex.match(components[1])))
            return "_".join(components[1:])


class EarthNet2021DataModule(pl.LightningDataModule):
    __TRACKS__ = {
        "iid": "iid_test_split/",
        "ood": "ood_test_split/",
        "ex": "extreme_test_split/",
        "sea": "seasonal_test_split/",
    }

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.hparams.update(vars(hparams))
        self.base_dir = Path(hparams.base_dir)

    @staticmethod
    def add_data_specific_args(parent_parser: Optional[Union[argparse.ArgumentParser, list]] = None):
        if parent_parser is None:
            parent_parser = []
        elif not isinstance(parent_parser, list):
            parent_parser = [parent_parser]

        parser = argparse.ArgumentParser(parents=parent_parser, add_help=False)

        parser.add_argument('--base_dir', type=str, default="data/datasets/")
        parser.add_argument('--test_track', type=str, default="iid")

        parser.add_argument('--noisy_masked_pixels', type=str2bool, default=True)
        parser.add_argument('--use_meso_static_as_dynamic', type=str2bool, default=True)
        parser.add_argument('--fp16', type=str2bool, default=False)
        parser.add_argument('--val_pct', type=float, default=0.05)
        parser.add_argument('--val_split_seed', type=int, default=42)

        parser.add_argument('--train_batch_size', type=int, default=1)
        parser.add_argument('--val_batch_size', type=int, default=1)
        parser.add_argument('--test_batch_size', type=int, default=1)
        parser.add_argument('--train_shuffle', type=str2bool, default=True)
        parser.add_argument('--pin_memory', type=str2bool, default=True)

        parser.add_argument('--train_workers', type = int, default = multiprocessing.cpu_count())
        parser.add_argument('--val_workers', type = int, default = 1)
        parser.add_argument('--test_workers', type = int, default = 1)

        parser.add_argument('--online_data_augmentation', type = str2bool, default = True)
        parser.add_argument('--online_time_downsample', type = int, default = 1)
        parser.add_argument('--time_downsample', type = int, default = 1)

        return parser

    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            earthnet_corpus = EarthNet2021Dataset(self.base_dir / "train",
                                                  noisy_masked_pixels=self.hparams.noisy_masked_pixels,
                                                  use_meso_static_as_dynamic=self.hparams.use_meso_static_as_dynamic,
                                                  fp16=self.hparams.fp16,
                                                  online_time_downsample=self.hparams.online_time_downsample,
                                                  time_downsample=self.hparams.time_downsample,
                                                  online_data_augmentation=self.hparams.online_data_augmentation)

            val_size = int(self.hparams.val_pct * len(earthnet_corpus))
            train_size = len(earthnet_corpus) - val_size

            self.earthnet_train, self.earthnet_val = random_split(earthnet_corpus, [train_size, val_size],
                                                                  generator=torch.Generator().manual_seed(
                                                                      int(self.hparams.val_split_seed)))

        if stage == 'test' or stage is None:
            self.earthnet_test = EarthNet2021Dataset(self.base_dir / self.__TRACKS__[self.hparams.test_track],
                                                     noisy_masked_pixels=self.hparams.noisy_masked_pixels,
                                                     use_meso_static_as_dynamic=self.hparams.use_meso_static_as_dynamic,
                                                     fp16=self.hparams.fp16,
                                                     online_time_downsample=self.hparams.online_time_downsample,
                                                     time_downsample=self.hparams.time_downsample,
                                                     online_data_augmentation=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_train, batch_size=self.hparams.train_batch_size,
                          shuffle=self.hparams.train_shuffle,
                          num_workers=self.hparams.train_workers, pin_memory=self.hparams.pin_memory, drop_last=True, 
                          persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_val, batch_size=self.hparams.val_batch_size,
                          num_workers=self.hparams.val_workers, pin_memory=self.hparams.pin_memory, 
                          persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.earthnet_test, batch_size=self.hparams.test_batch_size,
                          num_workers=self.hparams.test_workers, pin_memory=self.hparams.pin_memory, 
                          persistent_workers=True)
