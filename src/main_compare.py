import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torchvision

from net_module.net import ConvMultiHypoNet
from data_handle import data_handler as dh
from data_handle import dataset as ds

from util import utils_test
from util import zfilter

import pre_load

print("Program: animation SDD\n")

### Config
root_dir = Path(__file__).parents[1]

config_file1 = 'sdd_15_ewta_test.yml'
config_file2 = 'sdd_15_awta_test.yml'
config_file3 = 'sdd_15_swta_test.yml'
config_file_list = [config_file1, config_file2, config_file3]

data_from_zip = False
composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = ConvMultiHypoNet
PRED_OFFSET = 15

### Prepare
net_list = []
for config_f in config_file_list:
    dataset, _, net = pre_load.main_test_pre(root_dir, config_f, Dataset, data_from_zip, composed, Net)
    net_list.append(net)
num_net = len(config_file_list)

### Visualization option
idx_start = 50
idx_end = len(dataset)
pause_time = 0.1

### Visualize
fig, axes = plt.subplots(1, num_net)
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')
for idx in idc:

    [ax.cla() for ax in axes]

    hyposM_list = []
    for net in net_list:
        img, label, traj, index, hyposM, ref = pre_load.main_test(dataset, net, idx=idx)
        hyposM_list.append(hyposM)
    traj = np.array(traj)
    label_transpose = np.zeros_like(np.array(label))
    label_transpose[:,0] = label[:,1]
    label_transpose[:,1] = label[:,0]
    traj_transpose = np.zeros_like(traj)
    traj_transpose[:,0] = traj[:,1]
    traj_transpose[:,1] = traj[:,0]
    print(index, idx)

    ### Kalman filter
    P0 = zfilter.fill_diag((1,1,1,1))
    Q  = zfilter.fill_diag((0.1,0.1,0.1,0.1))
    R  = zfilter.fill_diag((0.1,0.1))
    KF_X, KF_P = utils_test.fit_KF(zfilter.model_CV, traj_transpose, P0, Q, R, PRED_OFFSET)

    title_list = ['EWTA', 'AWTA', 'SWTA']
    for hyposM, ax, title in zip(hyposM_list, axes, title_list):
        hypos_transpose = np.zeros_like(hyposM[0])
        hypos_transpose[:,0] = hyposM[0,:,1]
        hypos_transpose[:,1] = hyposM[0,:,0]
        hypos_clusters    = utils_test.fit_DBSCAN(hypos_transpose, eps=10, min_sample=3) # DBSCAN
        mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters) # Gaussian fitting
        ax.imshow(ref.transpose(0,1)) # show colored background
        # ax.imshow(img[-1,:].transpose(0,1), cmap='gray') # show gray-scale background
        utils_test.plot_Gaussian_ellipses(ax, mu_list, std_list)
        utils_test.plot_markers(ax, label_transpose[0], traj_transpose, hypos_transpose[np.newaxis,:,:], hypos_clusters)
        utils_test.set_axis(ax, title=title, y_label=title_list.index(title)==0)
    
    utils_test.plot_KF(axes[0], KF_X, KF_P)

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    if pause_time == 0:
        plt.pause(0.1)
        input()
    else:
        plt.pause(pause_time)

plt.show()


