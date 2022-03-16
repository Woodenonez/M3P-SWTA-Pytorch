import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torchvision

from net_module.net import ConvMultiHypoNet
from network_manager import NetworkManager
from data_handle import data_handler_zip as dh

from util import utils_test

print("Program: animation\n")

### Customize
past = 4
cpi = 2
dim_output = 2
num_components = 20
fc_input = 4608 # 98304, 386080

input_channel = (past+1)*cpi

model_path1 = os.path.join(Path(__file__).parents[1], 'Model/ewta_20m_20')
model_path2 = os.path.join(Path(__file__).parents[1], 'Model/awta_20m_20')
model_path3 = os.path.join(Path(__file__).parents[1], 'Model/swta_20m_20')
model_path_list = [model_path1, model_path2, model_path3]
name_list = ['EWTA', 'AWTA', 'SWTA']

zip_path  = os.path.join(Path(__file__).parents[1], 'Data/SID_Test_20.zip')
csv_path  = 'SID_Test_20/all_data.csv'
data_dir  = 'SID_Test_20/'

idx_start = 200
idx_end = 786
pause_time = 0

### Prepare data
composed = torchvision.transforms.Compose([dh.ToTensor()])
dataset = dh.ImageStackDataset(zip_path, csv_path, data_dir, channel_per_image=cpi, transform=composed)
print("Data prepared. #Samples:{}.".format(len(dataset)))
print('Sample: {\'image\':',dataset[0]['image'].shape,'\'label\':',dataset[0]['label'],'}')

### Initialize the model
mynet_list = []
for model_path in model_path_list:
    net = ConvMultiHypoNet(input_channel, dim_output, fc_input, num_components=num_components)
    myNet = NetworkManager(net, loss_function_dict={}, verbose=False)
    myNet.build_Network()
    myNet.model.load_state_dict(torch.load(model_path))
    myNet.model.eval() # with BN layer, must run eval first
    mynet_list.append(myNet)

### Visualize
fig, axes = plt.subplots(1,3)
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int') 
for idx in idc:

    [ax.cla() for ax in axes]
    
    img   = dataset[idx]['image']
    label = dataset[idx]['label'] * 1
    traj  = np.array(dataset[idx]['traj'])
    index = dataset[idx]['index']

    hypo_list = []
    for myNet in mynet_list:
        hyposM = myNet.inference(img)
        hypo_list.append(hyposM)

    ### DBSCAN - Density-Based Spatial Clustering of Applications with Noise
    cluster_list = []
    for hyposM in hypo_list:
        hypos_clusters = utils_test.fit_DBSCAN(hyposM[0], eps=0.5, min_sample=3)
        cluster_list.append(hypos_clusters)
    ###

    ### Gaussian fitting
    for i, hypos_clusters in enumerate(cluster_list):
        mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters) # Gaussian fitting
        for mu, std in zip(mu_list, std_list):
            patch = patches.Ellipse(mu, 3*std[0], 3*std[1], fc='y', zorder=0)
            axes[i].add_patch(patch)
    ###

    for ax, hyposM in zip(axes, hypo_list):
        utils_test.plot_on_simmap(ax, dataset[idx], hyposM[0,:,:])

    for i, ax in enumerate(axes):
        for j, hc in enumerate(cluster_list[i]):
            ax.scatter(hc[:,0], hc[:,1], marker='x', label=f"est{j+1}")
        ax.set_xlabel("x [m]", fontsize=14)
        ax.set_ylabel("y [m]", fontsize=14)
        ax.legend()
        ax.legend(prop={'size': 18}, loc='upper right')
        ax.axis('equal')
        ax.set_title(name_list[i], fontsize=24)

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    if pause_time == 0:
        plt.pause(0.1)
        input()
    else:
        plt.pause(pause_time)

plt.show()

