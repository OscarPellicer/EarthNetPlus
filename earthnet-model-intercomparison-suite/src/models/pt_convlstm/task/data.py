import argparse
import multiprocessing
import re, os
from typing import Union, Optional
import datetime
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import sys
from pathlib import Path
utils_dir = Path(__file__).resolve().parent.parent.parent.parent.parent.parent.parent
sys.path.append(str(utils_dir))
from utils import str2bool, load

landcover_dict= {0:'Clouds', 62:'Artificial surfaces and constructions', 73:'Cultivated areas', 75:'Vineyards', 
                 82:'Broadleaf tree cover', 83:'Coniferous tree cover', 102:'Herbaceous vegetation', 
                 103:'Moors and Heathland', 104:'Sclerophyllous vegetation', 105:'Marshes', 106:'Peatbogs', 
                 121:'Natural material surfaces', 123:'Permanent snow covered surfaces', 162:'Water bodies', 255:'No data'}

class EarthNet2021Dataset(Dataset):
    def __init__(self, folder: Union[Path, str], noisy_masked_pixels = False, 
                use_meso_static_as_dynamic = False, fp16 = False, use_unprocessed_data=False,
                online_data_augmentation=True, online_time_downsample=1, time_downsample=1.,
                climate_interp_order=False, land_cover_dir=None):
        if not isinstance(folder, Path):
            folder = Path(folder)
        assert (not {"target","context"}.issubset(set([d.name for d in folder.glob("*") if d.is_dir()])))
        self.filepaths = sorted(list(folder.glob(f"**/*.{'npz' if use_unprocessed_data else 'zip'}")))
        print(f'Found {len(self.filepaths)} samples')
        assert len(self.filepaths) > 0, f"No files found at: {str(folder)}"

        self.noisy_masked_pixels = noisy_masked_pixels
        self.use_meso_static_as_dynamic = use_meso_static_as_dynamic
        self.online_data_augmentation= online_data_augmentation
        self.online_time_downsample= online_time_downsample
        self.time_downsample= time_downsample
        self.use_unprocessed_data= use_unprocessed_data
        self.climate_interp_order= climate_interp_order
        self.land_cover_dir= land_cover_dir
        self.type = np.float16 if fp16 else np.float32
        self.rng = np.random.default_rng(seed=42)
        self.folder= folder

    def __getitem__(self, idx: int) -> dict:
        
        filepath = self.filepaths[idx]

        #For backwards compatibility
        if self.use_unprocessed_data:
            npz = np.load(filepath)

            #(x, y, c, t) -> (t, c, x, y) 
            highresdynamic = np.transpose(npz["highresdynamic"], (3, 2, 0, 1)).astype(self.type)
            highresstatic = np.transpose(npz["highresstatic"], (2, 0, 1)).astype(self.type)
            mesodynamic = np.transpose(npz["mesodynamic"], (3, 2, 0, 1)).astype(self.type)
            mesostatic = np.transpose(npz["mesostatic"], (2, 0, 1)).astype(self.type)

            if self.land_cover_dir != '':
                landcover= np.load(str(filepath).replace(str(self.folder), self.land_cover_dir).replace('context_', '').replace('context', ''))
                landcoverstatic= np.concatenate([landcover['landcover'] == k for k in list(landcover_dict.keys())[1:-1] ], axis=0)
                #landcoverstatic= np.transpose(landcover, (0, 1, 2)).astype(self.type)
            else: 
                landcoverstatic= np.nan

            masks = ((1 - highresdynamic[:, -1, :, :])[:, np.newaxis, :, :]).repeat(4, 1)

            images = highresdynamic[:, :4, :, :]

            images[np.isnan(images)] = 0
            images[images > 1] = 1
            images[images < 0] = 0
            mesodynamic[np.isnan(mesodynamic)] = 0
            highresstatic[np.isnan(highresstatic)] = 0
            mesostatic[np.isnan(mesostatic)] = 0

            if self.noisy_masked_pixels:
                images = np.transpose(images, (1, 0, 2, 3))
                all_pixels = images[np.transpose(masks, (1, 0, 2, 3)) == 1].reshape(4, -1)
                all_pixels = np.stack(int(images.size / all_pixels.size + 1) * [all_pixels], axis=1)
                all_pixels = all_pixels.reshape(4, -1)
                all_pixels = all_pixels.transpose(1, 0)
                np.random.shuffle(all_pixels)
                all_pixels = all_pixels.transpose(1, 0)
                all_pixels = all_pixels[:, :images.size // 4].reshape(*images.shape)
                images = np.where(np.transpose(masks, (1, 0, 2, 3)) == 0, all_pixels, images)
                images = np.transpose(images, (1, 0, 2, 3))

            if self.use_meso_static_as_dynamic:
                mesodynamic = np.concatenate([mesodynamic, mesostatic[np.newaxis, :, :, :].repeat(mesodynamic.shape[0], 0)], axis=1)

            #Perform climate time downsamplimg
            #mesodynamic= mesodynamic[...,39:41,39:41]
            t, c, h2, w2 = mesodynamic.shape
            mesodynamic = mesodynamic.reshape(t // 5, 5, c, h2, w2).mean(1)

            #Add date data
            _, _, w, h= highresdynamic.shape
            t2, _, _, _= mesodynamic.shape
            _, start_date, end_date, _, _, _, _, _, _, _, _ = self.__name_getter(filepath).split("_")
            end_day= datetime.datetime.strptime(end_date, '%Y-%m-%d').timetuple().tm_yday
            start_day= end_day - self.time_downsample * t2 * 5
            day_range= 2 * np.pi * np.arange(start_day, end_day-0.1, step=self.time_downsample*5) / 365.0  

            day_arr= np.ones((t2, 2, w, h)).astype(self.type)
            day_arr[:,0]*= np.sin(day_range)[:, None, None]
            day_arr[:,1]*= np.cos(day_range)[:, None, None]
            day_arr= day_arr.astype(self.type)

            #Better clim interpolation (alrady performed if preprocessed data is used!)
            if self.climate_interp_order > 0:
                _, _, w, h= highresdynamic.shape
                t2, c2, w2, h2= mesodynamic.shape
                if w2 != 80:
                    warnings.warn(f'To use better_climate_interp, mesodynamic must be 80x80; found {w2}x{h2}')
                else:
                    import cv2
                    #from skimage.transform import rescale # waaay slower!

                    assert self.climate_interp_order in [0,1,3], 'Only interpolation orders 0, 1, and 3 are supported'
                    interpolation= {0:cv2.INTER_NEAREST, 1:cv2.INTER_LINEAR, 3:cv2.INTER_CUBIC}[self.climate_interp_order]
                    cd= int(self.climate_interp_order/2) + 1 #1 extra to be safe (and for .5 fractions)
                    scale= 128/2
                    wb= int((2 + 2*cd)*scale)

                    mesodynamic_tp= np.transpose(mesodynamic, (2,3,1,0)).astype(np.float32)
                    mesodynamic_crop= mesodynamic_tp[39-cd:41+cd,39-cd:41+cd] #Get minimum crop size given the order
                    mesodynamic_crop= mesodynamic_crop.reshape(2 + 2*cd, 2 + 2*cd, t2*c2)
                    #mesodynamic_big= rescale(mesodynamic_crop, scale= 128/2, order=climate_interp_order, channel_axis=-1)
                    mesodynamic_big = cv2.resize(mesodynamic_crop, (wb, wb), interpolation=interpolation)
                    mesodynamic_res= mesodynamic_big[(wb-w)//2:(wb+w)//2, (wb-w)//2:(wb+w)//2]
                    mesodynamic= np.transpose(mesodynamic_res.reshape(h, w, c2, t2), (3,2,0,1)).astype(self.type)
            
        else:
            npz = load(str(filepath.resolve()))

            #(x, y, c, t) -> (t, c, x, y) 
            highresdynamic = np.transpose(npz["highresdynamic"],(3,2,0,1)).astype(self.type)[:, :4]
            highresstatic = np.transpose(npz["highresstatic"],(2,0,1)).astype(self.type)
            if self.climate_interp_order > 0:
                mesodynamic = np.transpose(npz["mesodynamic_interp"],(3,2,0,1)).astype(self.type)
            else: 
                mesodynamic = np.transpose(npz["mesodynamic"],(3,2,0,1)).astype(self.type)
            mesostatic = np.transpose(npz["mesostatic"],(2,0,1)).astype(self.type)
            landcoverstatic = np.transpose(npz["landcover"],(2,0,1)).astype(self.type)

            #highresdynamic= highresdynamic[:,[2,1,0,3]] #RGB -> GBR

            highresmask= np.transpose(npz["highresmask"],(2,0,1)).astype(self.type)
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
                warnings.warn('noisy_masked_pixels is no longer recommended', DeprecationWarning)

            if self.use_meso_static_as_dynamic:
                mesodynamic = np.concatenate([mesodynamic, mesostatic[np.newaxis, :, :, :].repeat(mesodynamic.shape[0], 0)], axis = 1)

            _, _, w, h= highresdynamic.shape
            t2, _, _, _= mesodynamic.shape

            #Standarization of coordinate data
            # min_long, max_long, min_lat, max_lat= -15, 30, 35, 70
            # coords= npz["coordinates"]
            # coordinates= ((coords[0] - min_long) / (max_long - min_long),
            #             (coords[1] - min_lat) / (max_lat - min_lat))
            # coords_arr= np.ones((t2, 2, w, h)).astype(self.type)
            # coords_arr[:,0]*= coords[0]
            # coords_arr[:,1]*= coords[1]

            #Cosine encoding of day of year
            if 'start_date' not in npz.keys(): #backwards compatibility
                end_day= npz['date'].timetuple().tm_yday
                start_day= end_day - self.time_downsample * t2 * 5
            else:
                end_day= npz['end_date'].timetuple().tm_yday
                start_day= end_day - self.time_downsample * t2 * 5
            day_range= 2 * np.pi * np.arange(start_day, end_day-0.1, step=self.time_downsample*5) / 365.0  

            day_arr= np.ones((t2, 2, w, h)).astype(self.type)
            day_arr[:,0]*= np.sin(day_range)[:, None, None]
            day_arr[:,1]*= np.cos(day_range)[:, None, None]
            day_arr= day_arr.astype(self.type)

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
            # images= np.mean(images.reshape( 
            #                 images.shape[0]//self.online_time_downsample, self.online_time_downsample,
            #                 *images.shape[1:]), axis=1)
            images= images[:: self.online_time_downsample]
            mesodynamic= np.mean(mesodynamic.reshape(
                            mesodynamic.shape[0]//self.online_time_downsample, 
                            self.online_time_downsample,
                            *mesodynamic.shape[1:]), axis=1)
            # masks= (np.mean(masks.reshape(
            #                 masks.shape[0]//self.online_time_downsample, 
            #                 self.online_time_downsample, 
            #                 *masks.shape[1:]), axis=1) > 0.49).astype(self.type)
            masks= masks[:: self.online_time_downsample]

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
            "scalars": [torch.from_numpy(day_arr),],
            "filepath": str(filepath),
            "cubename": self.__name_getter(filepath),
            # "long": coordinates[0],
            # "lat": coordinates[1],
            "dayofyear": start_day,
            "land_cover": landcoverstatic,
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
        "custom": "custom_test_split/",
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
        parser.add_argument('--use_unprocessed_data', type = str2bool, default = False)
        parser.add_argument('--climate_interp_order', type = int, default = 0)

        parser.add_argument('--land_cover_dir', type=str, default='')

        return parser

    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            earthnet_corpus = EarthNet2021Dataset(self.base_dir / "train",
                                                  noisy_masked_pixels=self.hparams.noisy_masked_pixels,
                                                  use_meso_static_as_dynamic=self.hparams.use_meso_static_as_dynamic,
                                                  fp16=self.hparams.fp16,
                                                  online_time_downsample=self.hparams.online_time_downsample,
                                                  time_downsample=self.hparams.time_downsample,
                                                  online_data_augmentation=self.hparams.online_data_augmentation,
                                                  use_unprocessed_data=self.hparams.use_unprocessed_data,
                                                  climate_interp_order=self.hparams.climate_interp_order,
                                                  land_cover_dir= os.path.join(self.hparams.land_cover_dir, "train") 
                                                    if self.hparams.land_cover_dir != "" else "")

            val_size = int(self.hparams.val_pct * len(earthnet_corpus))
            train_size = len(earthnet_corpus) - val_size

            self.earthnet_train, self.earthnet_val = random_split(earthnet_corpus, [train_size, val_size],
                                                                  generator=torch.Generator().manual_seed(
                                                                      int(self.hparams.val_split_seed)))

        if stage == 'test' or stage is None:
            if self.hparams.use_unprocessed_data:
                track_dir= self.base_dir / self.__TRACKS__[self.hparams.test_track] / 'context'
            else:
                track_dir= self.base_dir / self.__TRACKS__[self.hparams.test_track]
            self.earthnet_test = EarthNet2021Dataset(track_dir,
                                                     noisy_masked_pixels=self.hparams.noisy_masked_pixels,
                                                     use_meso_static_as_dynamic=self.hparams.use_meso_static_as_dynamic,
                                                     fp16=self.hparams.fp16,
                                                     online_time_downsample=self.hparams.online_time_downsample,
                                                     time_downsample=self.hparams.time_downsample,
                                                     online_data_augmentation=False,
                                                     use_unprocessed_data=self.hparams.use_unprocessed_data,
                                                     climate_interp_order=self.hparams.climate_interp_order,
                                                     land_cover_dir= os.path.join(self.hparams.land_cover_dir, self.__TRACKS__[self.hparams.test_track]) 
                                                        if self.hparams.land_cover_dir != "" else "")

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
