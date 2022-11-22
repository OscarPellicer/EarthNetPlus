# EarthNet+
Some changes have been made wrt [EarhNet toolkit](https://github.com/earthnet2021/earthnet-toolkit) and [TUM's conv LSTM approach](https://github.com/dcodrut/weather2land):
 - Improved expermient management: all experiment data (tensorboard logs, model predictions, final scores) is now stored by default in the experiment folder.
 - Improved configuration / train / test scripts
 - Improved preprocessing: if using a preprocessed dataset (read more below), missing data (e.g. due to clouds) now is reconstruced by using previous and next available images in the time series.
 - Improved preprocessing: The data is preprocessed and stored to occupy less than half the original space for quicker reads during model training.
 - Improved preprocessing: Climate variables are interpolated (0th, 1st, or 3rd order)
 - You can now also add landcover masks to the inputs
 - Added option: more convolutions per layer
 - Added option: adding day of year data sin-cos-encoded
 - Added option: online temporal downscaling
 - Added option: simple online data augmentation (90º rotations and flipping)
 - No more warnings or errors during training in the latest pytorch lightning version.
 - Several more small fixes and QoL upgrades

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

You can choose to run the model feeding it the EarthNet data directly, or to first preprocess the data. 
Both options are similar in terms of functinality, but by first preprocessing the data, you can offload some computation from the data loader.
Additionally, the (very simple) gap filler can only be used with preprecessed data (it would be very slow for online processing).
Note that some things may have not been tested in both scenarios, and might fail upon training. Open an issue or contact me if in doubt!
In the next command, you can choose to download the data (or not) and to preprocess it (or not).

```{bash}
conda activate earthnet

cd ~/earthnet
python preprocess.py --download True --preprocess True --download_path "D:\EarthNet" --preprocess_path "D:\EarthNet_pre"
```

## Training, testing and evaluation

First, edit the file `./earthnet-model-intercomparison-suite/src/models/pt_convlstm/configs/conv_lstm.yaml` to point to the correct paths
There are three configuration files: 
 - `conv_lstm_legacy.yaml` trains a model that should be very close to the original in [TUM's conv LSTM approach](https://github.com/dcodrut/weather2land), for reference
 - `conv_lstm_test.yaml` adds some functionality that has been tested to improve predicions

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