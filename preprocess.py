from pathlib import Path
import os, sys, numpy as np, matplotlib.pyplot as plt, glob, tqdm
import datetime
from utils import timer, save, load, quick_fill, ESA_scenes, detect_outlier_slices, plot_all
from argparse import ArgumentParser
from random import Random
from scipy.ndimage import binary_dilation

sys.path.append('earthnet-toolkit')
from earthnet import Downloader, get_coords_from_cube

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def preprocess_dataset(
    download_path,
    preprocessed_path, 
    time_downsample= 1, #Only 1, 2 or 5 are acceptable for an (almost) proper earthnet evaluation
    process_target= False, #If False, target folder in test tracks is ignored (preferred)
    first_forward= False, #Only apply forward filling
    plot_results= 100,
    overwrite= True,
    save_cubes= True,
    masking_threshold_low= 0.075,
    masking_threshold_high= 0.4,
    subsets= ['iid_test_split', 'train', 'ood_test_split', 'seasonal_test_split', 'extreme_test_split'],
    ):
    '''
        Preprocesses the dataset in the following ways:
            - Quality mask is slighlty processed
            - Missing data is filled from previous and next slices
            - Only the center 2x2 crop of climatic maps is stored
            - Data is stored and compressed into a pickled dictionary saved as a .zip
            - Optionally, data is temporarily subsampled
            - Coordinate and date data is added to the final dictionary
    '''
    se= np.zeros((3,3,3)); se[1]=1; se=se.T
    image_dict= {}

    for subset in subsets:
        print(f'{subset=}')
        if subset=='train':
            files= glob.glob(os.path.join(download_path, subset, '*', '*.npz'))
        else:
            files= glob.glob(os.path.join(download_path, subset, 'context', '*', '*.npz'))
        Random(42).shuffle(files)

        for i, cubepath in enumerate( tqdm.tqdm(files) ):
            #if not i in [13,33]: continue    
            cubename= os.path.basename(cubepath).split('.')[0]
            new_path= cubepath.replace(download_path, preprocessed_path).replace('.npz', '').replace('context_', '').replace('context', '')
            if os.path.exists(new_path  + '.zip') and not overwrite: continue

            if subset=='train' or not process_target:
                sample = np.load(cubepath)
                hrdyn= sample["highresdynamic"]
            else:
                sample= np.load(cubepath)
                target= np.load(cubepath.replace('context', 'target'))
                hrdyn= np.concatenate([sample["highresdynamic"], target["highresdynamic"]], axis=-1)

            img= hrdyn[...,[2,1,0,3], :]

            img[np.isnan(img)] = 0
            img[img > 1] = 1
            img[img < 0] = 0

            #Homogeneous processing
            qmask= (hrdyn[...,-1,:]>0.5) | np.all(hrdyn[...,0,:] == 0, axis=(0,1), keepdims=True) | np.isnan(hrdyn[...,0,:])
            qmask= binary_dilation(qmask, se, 3) #Extend mask just in case

            slice_mask_cover= qmask.mean(axis=(0,1))
            qmask[..., slice_mask_cover >= masking_threshold_high]= 1 #Just mask the whole image if it is very degraded
            qmask[..., slice_mask_cover < masking_threshold_low]= 0 #Just ignore the clouds if there are very few of them

            #Now, fix cs for missing images via ffilling + bfilling
            qmask= np.repeat(qmask[...,None,:], repeats=img.shape[2], axis=2) #Extend qmask to the same shape as img
            img, qmask, ffill, fmask, bfill, bmask= quick_fill(img, qmask, first_forward=first_forward)

            #Finally, sample using the median every N images
            img= np.mean(img.reshape(*img.shape[:-1], img.shape[-1]//time_downsample, time_downsample), axis=-1)
            qmask= np.mean(qmask.reshape(*qmask.shape[:-1], qmask.shape[-1]//time_downsample, time_downsample), axis=-1) > 0.499

            #Build data dict
            image_dict['highresdynamic']= img #RGB + NIR
            image_dict['highresstatic']= sample["highresstatic"] #DEM
            image_dict['highresmask']= qmask[...,0,:] #Bad quality mask
            image_dict['mesostatic']= sample["mesostatic"] #DEM

            md= sample["mesodynamic"]
            meso_down_factor= time_downsample * 5
            md= np.mean(md.reshape(*md.shape[:-1], md.shape[-1]//meso_down_factor, meso_down_factor), axis=-1)
            image_dict['mesodynamic']= md #Wheather features
            image_dict['mesodynamic']= image_dict['mesodynamic'][39:41,39:41]

            #Get coordinates
            coords= get_coords_from_cube(cubename, return_meso=False, ignore_warning=True, train=subset=='train')
            image_dict['coordinates']= ((coords[0] + coords[2])/2, (coords[1] + coords[3])/2)

            #Get start date
            if subset=='train':
                _, start_date, end_date, _, _, _, _, _, _, _, _ = os.path.splitext(cubename)[0].split("_")
            else:
                _, _, start_date, end_date, _, _, _, _, _, _, _, _ = os.path.splitext(cubename)[0].split("_")
            #image_dict['date']= datetime.datetime.strptime(end_date, '%Y-%m-%d') #For backwards compatibilty
            image_dict['start_date']= datetime.datetime.strptime(start_date, '%Y-%m-%d')
            image_dict['end_date']= datetime.datetime.strptime(end_date, '%Y-%m-%d')

            #Save
            if save_cubes:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                save(new_path, image_dict, 'zip')
            #ndvi= (img[...,3,:] - img[...,2,:]) / (img[...,3,:] + img[...,2,:] + 1e-6)

            #Plot results
            if i < plot_results:
                #h, w, c, t -> t, c, h, w
                hrdyn, img, qmask= np.transpose(hrdyn, (3,2,0,1)), np.transpose(img, (3,2,0,1)), np.transpose(qmask, (3,2,0,1))
                plot_all(hrdyn, img[:,[2,1,0]], qmask, downsample_time=1, title=f'i={i}: {cubename}', scale=1, 
                         save_path=os.path.join(preprocessed_path, subset + '_samples', cubename + '.png'))
            
            
if __name__ == '__main__':    
    parser = ArgumentParser()
    parser.add_argument('--download', type=str2bool, default=False, help='Whether to download the data')
    parser.add_argument('--download_path', type=str, required=True, metavar='path/where/data/is/to/be/downloaded', help='Download path')
    parser.add_argument('--preprocess_path', type=str, required=True, metavar='path/where/preprocessed/data/will/be/saved',
                        help='Path for preprocessed data')
    parser.add_argument('--overwrite', type=str2bool, default=False, help='Whether to overwrite already processed data')
    args = parser.parse_args()
    
    if args.download:
        Downloader.get(args.download_path, 'all')
    preprocess_dataset(args.download_path, args.preprocess_path, overwrite=args.overwrite)