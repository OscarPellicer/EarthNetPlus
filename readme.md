# EarthNet+
Some changes have been made wrt [EarhNet toolkit](https://github.com/earthnet2021/earthnet-toolkit) and [TUM's conv LSTM approach](https://github.com/dcodrut/weather2land):
 - Improved expermient management: all experiment data (tensorboard logs, model predictions, final scores) is now stored by default in the experiment folder.
 - Improved configuration / train / test scripts
 - Improved preprocessing: missing data (e.g. due to clouds) is now reconstruced by using previous and next available images in the time series.
 - Improved preprocessing: The data is preprocessed and stored to occupy less than half the original space for quicke reads during model training.
 - More model configuration options: added options for more convs per layer, for adding location and day of year data, for applying online temporal downscaling, etc., added data augmentation
 - No more warnings or errors during training in the latest pytorch lightning version.
 - Several more small fixes and qol upgrades

## Installation
``` {bash}
mkdir ~/earthnet
cd ~/earthnet
conda create --name earthnet python=3.8
conda activate earthnet

conda install numpy scipy scikit-image pandas jupyter git matplotlib pillow shapely scikit-learn imgaug scipy seaborn geopandas
conda install tqdm fire pyproj imantics opencv netCDF4 ffmpeg sk-video -c conda-forge
conda install pytorch torchvision torchaudio cudatoolkit=11.6 segmentation-models-pytorch pytorch-lightning -c pytorch -c conda-forge

git clone [this repo.git]
```

## First time setup
```{bash}
conda activate earthnet

cd ~/earthnet
python preprocess.py --download True --download_path "D:\EarthNet" --preprocess_path "D:\EarthNet_pre"
```

## Training, testing and evaluation
First, edit the file `./earthnet-model-intercomparison-suite/src/models/pt_convlstm/configs/conv_lstm.yaml` to point to the correct paths
```{bash}
conda activate earthnet
cd "./earthnet-model-intercomparison-suite/src/models/pt_convlstm"

python train.py --setting ".\configs\conv_lstm.yaml"
python test.py --setting "D:\oscar\Earthnet experiments\conv_lstm\version_25\settings.yaml" --version 25 --track all
python evaluate.py --setting "D:\oscar\Earthnet experiments\conv_lstm\version_25\settings.yaml" --version 25 --track all
```

You can chain commands toghether like this:
```{bash}
python test.py --setting "D:\oscar\Earthnet experiments\conv_lstm\version_25\settings.yaml" --version 25 --track all && python evaluate.py --setting "D:\oscar\Earthnet experiments\conv_lstm\version_25\settings.yaml" --version 25 --track all
```
where:
 - `&` (win) or `;` (linux): Execute after first command
 - `&&`: Execute if first command was successful
 - `||`: Execute if first command was unsuccessful
 
Or you simply call `test.py` with the evaluate argument set to true:
```{bash}
python test.py --setting "D:\oscar\Earthnet experiments\conv_lstm\version_25\settings.yaml" --version 25 --track all --evaluate True
```