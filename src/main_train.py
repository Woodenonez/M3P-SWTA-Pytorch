import os, sys
from pathlib import Path

import torchvision

from net_module import loss_functions as loss_func
from net_module.net import ConvMultiHypoNet
from data_handle import data_handler as dh
from data_handle import dataset as ds

import pre_load

print("Program: training\n")

TRAIN_MODEL = 'ewta' # ewta, awta, swta

### Config
root_dir = Path(__file__).parents[1]
config_file = f'sdd_15_{TRAIN_MODEL}.yml'

if TRAIN_MODEL == 'ewta':
    meta_loss = loss_func.meta_loss
    k_top_list = [20]*10 + [10]*10 + [8]*10 + [7]*10 + [6]*10 + [5]*10 + [4]*10 + [3]*10 + [2]*10 + [1]*10
elif TRAIN_MODEL == 'awta':
    meta_loss = loss_func.ameta_loss
    k_top_list = [20]*10 + [10]*10 + [8]*10 + [7]*10 + [6]*10 + [5]*10 + [4]*10 + [3]*10 + [2]*10 + [1]*10
elif TRAIN_MODEL == 'swta':
    meta_loss = loss_func.ameta_loss
    k_top_list = [20]*10 + [10]*10 + [8]*5 + [7]*5 + [6]*5 + [5]*10 + [4]*10 + [3]*10 + [2]*10 + [1]*10 + [0]*15
else:
    raise ModuleNotFoundError(f'Cannot find mode {TRAIN_MODEL}.')

loss_dict = {'meta':meta_loss, 'base':loss_func.loss_mse, 'metric':None}

data_from_zip = False
composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = ConvMultiHypoNet

### Training
pre_load.main_train(root_dir, config_file, Dataset=Dataset, Net=Net, 
                    zip=data_from_zip, transform=composed, loss=loss_dict, k_top_list=k_top_list, num_workers=2)
