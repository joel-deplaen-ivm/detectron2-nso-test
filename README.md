# Substation detection on NSO imagery

Pipeline for training data creation, training and inference of a MaskRCNN to detect electyricxal substations in the Netherlands using NSO Superview - 0.5 m resolution

## 0. Installation

### Instalation guide .yml

**requirements.yml**

**Procedure torch, torchvision, detectron2**

pip3 install \
torch==1.10.2 \
torchvision==0.11.3 \
--extra-index-url https://download.pytorch.org/whl/cu113

python -m pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

On cluster:

module load 2022

        module load CUDA/11.3.1

### Test environment by importing:

import detectron2         # detectron2
import torch              # pytorch
import cv2 as cv          # openCV
import numpy as np        # numpy
from osgeo import gdal    # GDAL
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

### Verify torch, torchvision, cuda compatibility by running:

python -m detectron2.utils.collect_env

See: https://stackoverflow.com/questions/70831932/cant-connect-to-gpu-when-building-pytorch-projects

or

python -c "import uutils; uutils.torch_uu.gpu_test()

see: https://stackoverflow.com/questions/66992585/how-does-one-use-pytorch-cuda-with-an-a100-gpu

## 1. Data Preperation

Aim at the preperation of the imagery andf annotation for DL training

### 1.1 extract_osm

**Extract and filter OSM data for electrical substations**

Check: *subs_detection/notebooks/prepare_substation.ipynb*

The functions can be found in *subs_detection/utils/*

### 1.2 tiling_nso

**Create tiles of satelite imagery and annotation for DL model training**

Check: *subs_detection/notebooks/tiling_nso.ipynb*

### Note: osmconf.ini

Should be added to overide the gdal .ini file in conda env:

Also in *subs_detection/scripts/extract_osm_sub.py*: 

gdal.SetConfigOption("OSM_CONFIG_FILE", os.path.join('..',"osmconf.ini"))"

### 1.3 convert_tif_nso

### 1.4 create_jsons_nso

### 1.5 plot_truth_coordinates

## 2 Train Model

### 2.1 train_net

## 3. Run Model

## 3.1 inference