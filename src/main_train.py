import os, sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

# 1. Architecture
from net_module.net import ConvMultiHypoNet
# 2. Training manager
from network_manager import NetworkManager
# 3. Loss functions
from net_module import loss_functions as loss_func
# 4. Data handler
from data_handle import data_handler_zip as dh

from util import utils_yaml

import pickle
from datetime import datetime

print("Program: training\n")
if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
        #   'Current:', torch.cuda.current_device(), torch.cuda.get_device_name(0))
else:
    print(f'CUDA not working! Pytorch: {torch.__version__}.')
    sys.exit(0)
torch.cuda.empty_cache()

### Config file name
config_file = 'awta_20.yml'
# loss_dict = {'meta':loss_func.meta_loss,  'base':loss_func.loss_mse, 'metric':None} # for EWTA
loss_dict = {'meta':loss_func.ameta_loss, 'base':loss_func.loss_mse,  'metric':None} # for AWTA and SWTA
k_top_list = [20]*2 + [10]*2 + [8]*2 + [7]*2 + [6]*2 + [5]*2 + [4]*2 + [3]*2 + [2]*2 + [1]*2 # SID-EWTA/AWTA, 20 epochs
# k_top_list = [20]*2 + [10]*2 + [8]*1 + [7]*1 + [6]*1 + [5]*2 + [4]*2 + [3]*2 + [2]*2 + [1]*2 + [0]*3 # SID-SWTA, 20 epochs

### Load parameters and define paths
root_dir = Path(__file__).parents[1]
param_path = os.path.join(root_dir, 'Config/', config_file)
param = utils_yaml.from_yaml(param_path)

save_path = os.path.join(root_dir, param['model_path'])

zip_path  = os.path.join(root_dir, param['zip_path'])
csv_path  = os.path.join(param['data_name'], param['label_csv'])
data_dir  = param['data_name']
print("Save to", save_path)

### Prepare data
composed = torchvision.transforms.Compose([dh.ToTensor()])
dataset = dh.ImageStackDataset(zip_path,csv_path, data_dir, channel_per_image=param['cpi'], transform=composed)
myDH = dh.DataHandler(dataset, batch_size=param['batch_size'], validation_prop=param['validation_prop'], validation_cache=param['batch_size'])
print("Data prepared. #Samples(training, val):{}, #Batches:{}".format(myDH.return_length_ds(), myDH.return_length_dl()))
print('Sample: {\'image\':',dataset[0]['image'].shape,'\'label\':',dataset[0]['label'],'}')

### Initialize the model
net = ConvMultiHypoNet(param['input_channel'], param['dim_out'], param['fc_input'], num_components=param['num_components'])
myNet = NetworkManager(net, loss_dict, early_stopping=param['early_stopping'], device=param['device'])
myNet.build_Network()
model = myNet.model

### Training
start_time = time.time()
myNet.train(myDH, param['batch_size'], param['epoch'], k_top_list=k_top_list, val_after_batch=10)
total_time = round((time.time()-start_time)/3600, 4)
if (save_path is not None) & myNet.complete:
    torch.save(model.state_dict(), save_path)
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\nTraining done: {} parameters. Cost time: {}h.".format(nparams, total_time))

### Visualize the training process
now = datetime.now()
dt = now.strftime("%d_%m_%Y__%H_%M_%S")

myNet.plot_history_loss()
plt.savefig(dt+'.png', bbox_inches='tight')
plt.close()

loss_dict = {'loss':myNet.Loss, 'val_loss':myNet.Val_loss}
with open(dt+'.pickle', 'wb') as pf:
    pickle.dump(loss_dict, pf)
