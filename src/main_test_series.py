import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision

# 1. Architecture
from net_module.net import ConvMultiHypoNet
# 2. Training manager
from network_manager import NetworkManager
# 3. Data handler
from data_handle import data_handler as dh

from util import utils_yaml
from util import utils_test

print("Program: animation\n")

### Config file name
config_file = 'awta_1t10_test.yml'

### Load parameters and define paths
root_dir = Path(__file__).parents[1]
param_path = os.path.join(root_dir, 'Config/', config_file)
param = utils_yaml.from_yaml(param_path)
param['device'] = 'multi'
param['data_name'] = 'SID_Show_10'

model_path = os.path.join(root_dir, param['model_path'])
data_dir   = os.path.join(root_dir, param['data_root'], param['data_name'])
csv_path   = os.path.join(root_dir, param['data_root'], param['data_name'], param['label_csv'])
print("Load from", model_path)

### Visualization option
idx_start = 0
idx_end = 786
pause_time = 0
pause_time = 0

### Prepare data
composed = torchvision.transforms.Compose([dh.ToTensor()])
dataset = dh.ImageStackDataset(csv_path=csv_path, root_dir=data_dir, channel_per_image=param['cpi'], transform=composed, T_channel=param['with_T'])
print("Data prepared. #Samples:{}.".format(len(dataset)))
print('Sample: {\'image\':',dataset[0]['image'].shape,'\'label\':',dataset[0]['label'],'}')

# fig, axes = plt.subplots(5,5)
### Initialize the model
net = ConvMultiHypoNet(param['input_channel'], param['dim_out'], param['fc_input'], num_components=param['num_components'])
myNet = NetworkManager(net, loss_function_dict={}, verbose=False, device=param['device'])
myNet.build_Network()
myNet.model.load_state_dict(torch.load(model_path))
myNet.model.eval() # with BN layer, must run eval first

### Visualize
fig, ax = plt.subplots()
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')
for idx in idc:

    plt.cla()
    
    img = dataset[idx]['image']

    img_list = []
    for i in range(1,11):
        img_temp = torch.ones_like(img[0,:,:]) * i
        img_temp = torch.cat((img[:-1,:,:], img_temp.unsqueeze(0)), dim=0)
        img_list.append(img_temp)

    hyposM_list = []
    for im in img_list:
        hyposM = myNet.inference(im) # BxMxC
        hyposM_list.append(hyposM)

    ### DBSCAN - Density-Based Spatial Clustering of Applications with Noise
    hypos_clusters_list = []
    for hyposM in hyposM_list:
        hypos_clusters = utils_test.fit_DBSCAN(hyposM[0], eps=0.5, min_sample=3) # DBSCAN
        hypos_clusters_list.append(hypos_clusters)
    ###

    # utils_test.plot_parzen_distribution(ax, hyposM[0,:,:])

    ### Gaussian fitting
    # mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters) # Gaussian fitting
    ###

    utils_test.plot_on_simmap(ax, dataset[idx], hyposM_list[0][0,:,:])
    for hyM in hyposM_list[1:]:
        plt.scatter(hyM[:,:,0], hyM[:,:,1], c='r', marker='.')
    
    color_ref = ['r','b','g','y','k','k','k','k']
    for hypos_clusters in hypos_clusters_list:
        for i, hc in enumerate(hypos_clusters):
            plt.scatter(hc[:,0], hc[:,1], marker='x', c=color_ref[i])
            # plt.scatter(hc[:,0], hc[:,1], marker='x', label=f"est{i+1}")
    plt.xlabel("x [m]", fontsize=14)
    plt.ylabel("y [m]", fontsize=14)
    plt.legend()
    ax.axis('equal')
    plt.legend(prop={'size': 14}, loc='upper right')

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    if pause_time == 0:
        plt.pause(0.1)
        input()
    else:
        plt.pause(pause_time)

plt.show()

