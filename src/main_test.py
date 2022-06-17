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

print("Program: animation\n")

### Config
root_dir = Path(__file__).parents[1]
config_file = 'sdd_15_swta_test.yml'

data_from_zip = False
composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = ConvMultiHypoNet
PRED_OFFSET = 15

### Prepare
dataset, _, net = pre_load.main_test_pre(root_dir, config_file, Dataset, data_from_zip, composed, Net)

### Visualization option
idx_start = 0
idx_end = len(dataset)
pause_time = 0.1

### Visualize
fig, ax = plt.subplots()
idc = np.linspace(idx_start, idx_end, num=idx_end-idx_start).astype('int')
for idx in idc:

    plt.cla()
    
    img, label, traj, index, hyposM, ref = pre_load.main_test(dataset, net, idx=idx)
    traj = np.array(traj)

    ### Kalman filter
    P0 = zfilter.fill_diag((1,1,1,1))
    Q  = zfilter.fill_diag((0.1,0.1,0.1,0.1))
    R  = zfilter.fill_diag((0.1,0.1))
    KF_X, KF_P        = utils_test.fit_KF(zfilter.model_CV, traj, P0, Q, R, PRED_OFFSET)

    ### CGF
    hypos_clusters    = utils_test.fit_DBSCAN(hyposM[0], eps=20, min_sample=3) # DBSCAN
    mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters) # Gaussian fitting
    
    ### Vis
    ax.imshow(img[1,:,:], cmap='gray')
    utils_test.plot_Gaussian_ellipses(ax, mu_list, std_list)
    utils_test.plot_markers(ax, label[0], traj, hyposM, hypos_clusters)
    utils_test.plot_KF(ax, KF_X, KF_P)
    utils_test.set_axis(ax, title='WTA')

    if idx == idc[-1]:
        plt.text(5,5,'Done!',fontsize=20)
    if pause_time == 0:
        plt.pause(0.1)
        input()
    else:
        plt.pause(pause_time)

plt.show()

