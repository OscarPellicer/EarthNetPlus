from scipy.ndimage import binary_dilation
from scipy.ndimage.morphology import binary_erosion
from random import Random
import tqdm
from time import time
import pickle, bz2, lzma, gzip
import numpy as np
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

se= np.zeros((3,3,3)); se[1]=1; se=se.T

ESA_scenes= {0: 'No data', 1: 'Saturated / Defective', 2: 'Dark Area Pixels',
             3: 'Cloud Shadows', 4: 'Vegetation', 5: 'Bare Soils', 6: 'Water',
             7: 'Clouds low probability / Unclassified', 8: 'Clouds medium probability',
             9: 'Clouds high probability', 10: 'Cirrus', 11: 'Snow / Ice'}

def timer(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r}\n - Executed in {(t2-t1):.4f}s\n - KWargs: {kwargs}')
        return result
    return wrap_func

protocol_dict= {'lzma':lzma.open, 'bz2':bz2.BZ2File, 
                'zip': gzip.open, 'none': open}
def save(name, data, protocol='lzma'):
    opener= protocol_dict[protocol]
    with opener(name + '.' + protocol, 'wb') as f: 
        pickle.dump(data, f)
        
def load(name):
    opener= protocol_dict[name.split('.')[-1]]
    with opener(name, 'rb') as f:
        return pickle.load(f)

def quick_fill(img, qmask, first_forward=True):
    '''
        Fills spots marked by qmask by performing forward filling wrt 
        last dimension (assumed to be the time dimension), and then
        backwards filling. Then takes the average of both fillings
        shape= (h, w, c, t)
    '''
    assert np.all(img.shape == qmask.shape), 'Input arrrays must have the same shape'

    #Forward fill until everything is filled
    ffill, fmask, fmean= img.copy(), qmask.copy(), 1
    while fmask.mean() != fmean:
        #plot(ffill[...,[0,1,2],1], masks=[fmask[...,0,1]], is_color=True)
        fmean= fmask.mean() 
        ffill[...,1:]= np.where(fmask[...,1:] & ~fmask[...,:-1], ffill[...,:-1], ffill[...,1:])
        fmask[...,1:]&= fmask[...,:-1]

    #Backwards fill until everything is filled
    if first_forward:
        bfill, bmask, bmean= ffill.copy(), fmask.copy(), 1
    else:
        bfill, bmask, bmean= img.copy(), qmask.copy(), 1
        
    while bmask.mean() != bmean:
        #plot(bfill[...,[0,1,2],-3], masks=[bmask[...,0,-3]], is_color=True)
        bmean= bmask.mean()
        bfill[...,:-1]= np.where(bmask[...,:-1] & ~bmask[...,1:], bfill[...,1:], bfill[...,:-1])
        bmask[...,:-1]&= bmask[...,1:]

    #Combine bfill with ffill
    img= (bfill*~bmask + ffill*~fmask)/2
    img[bmask | fmask]*= 2
    
    #Now fill the last missing data with the average value of a time slice
    img= np.where(bmask & fmask, img.mean(axis=(0,1), keepdims=True), img)
    
    return img, qmask, ffill, fmask, bfill, bmask

def detect_outlier_slices(img):
    '''
        To do: make proper statistical test WIP
        shape= (h, w, c, t)
    '''
    nan_slices= np.isnan(img).all(axis=(0,1,2))
    mean= np.nanmean(img, axis=(0,1,2))
    std= np.nanstd(img, axis=(0,1,2))
    grand_mean= np.nanmean(mean)
    grand_std= np.nanstd(std)
    print(mean, std, grand_mean, grand_std)
    
    z_values= (mean - grand_mean) / grand_std
    return nan_slices | (np.abs(z_values) > 2), z_values
    

def plot_all(y1, y2=None, mask=None, downsample_time=1, t2=0, title=None, downsample_space=2,
             spacing=(20, 20), scale=1, channels=[2,1,0], perc=99, dpi=300, save_path=None, **kwargs):
    '''
        y1, y2, and mask must have shapes: (t, c, h, w)
        mask equals 1 for bad pixels
    '''    
    t, c, h, w= y1.shape
    t3= t2 if t2 != 0 else t
    y_stack= np.reshape( np.transpose(y1[-t2::downsample_time], (0,2,3,1)), (w*t3//downsample_time, w, c) )
    
    if y2 is not None:
        tb, cb,_, _= y2.shape
        yp_stack= np.reshape( np.transpose(y2[-t2::downsample_time], (0,2,3,1)), (w*t3//downsample_time, w, cb) )
        y_concat= np.transpose(np.concatenate([y_stack[...,channels], yp_stack[...,channels]], axis=1), (1,0,2))
    else: 
        y_concat= np.transpose(y_stack[...,channels], (1,0,2))
        
    if mask is not None:
        _, cm, _, _= mask.shape
        m_stack= np.reshape( np.transpose(mask[-t2::downsample_time], (0,2,3,1)), (w*t3//downsample_time, w, cm) )
        m_concat= np.transpose(np.concatenate([m_stack, np.zeros_like(m_stack)], axis=1), (1,0,2))
    else:
        m_concat= np.zeros_like(y_concat)

    img_to_plot= np.flip(y_concat.astype(np.float32), axis=0)
    m_to_plot= np.flip(m_concat[...,0], axis=0) > 0.5

    valid_pixels= img_to_plot[~m_to_plot] if mask is not None else img_to_plot
    img_to_plot= np.minimum(np.maximum(img_to_plot/(np.percentile(valid_pixels, perc) - \
                                                    np.percentile(valid_pixels, 100-perc)), 0), 1)
    
    if downsample_space > 1:
        img_to_plot= img_to_plot[::downsample_space, ::downsample_space]
        m_to_plot= m_to_plot[::downsample_space, ::downsample_space]
        
    m_to_plot^= binary_erosion(m_to_plot, np.ones((3,3))).astype(m_to_plot.dtype)
    color= [1,0.1,0.1] #Red
    for i in range(3): img_to_plot[m_to_plot,i]= color[i]
    
    plt.rc('font', size=4) 
    f= plt.figure()#figsize=( t3//downsample_time*4*scale, 8*scale))
    plt.imshow(img_to_plot, **kwargs)
    plt.title(title)
    plt.tight_layout()
    plt.axis('off')
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, format=save_path.split('.')[-1], bbox_inches='tight')
        plt.close(f)
    else: 
        plt.show()