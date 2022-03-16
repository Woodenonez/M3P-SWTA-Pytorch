import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import torch
import torchvision

# 1. Architecture
from net_module.net import ConvMultiHypoNet, ConvMixtureDensityNet
# 2. Training manager
from network_manager import NetworkManager
# 3. Data handler
from data_handle import data_handler_zip as dh

from util import utils_test
from util import utils_yaml
from util import zfilter

print("Program: animation\n")

### Config file name
# config_file = 'mdn_20_test.yml'
config_file = 'ewta_20_test.yml'

### Load parameters and define paths
root_dir = Path(__file__).parents[1]
param_path = os.path.join(root_dir, 'Config/', config_file)
param = utils_yaml.from_yaml(param_path)
param['device'] = 'cuda' # cuda or multi

model_path = os.path.join(root_dir, param['model_path'])

zip_path  = os.path.join(root_dir, param['zip_path'])
csv_path  = os.path.join(param['data_name'], param['label_csv'])
data_dir  = param['data_name']
print("Save to", model_path)

### Visualization option
idx_start = 0
idx_end = 786
pause_time = 0.1

### Prepare data
composed = torchvision.transforms.Compose([dh.ToTensor()])
dataset = dh.ImageStackDataset(zip_path, csv_path, data_dir, channel_per_image=param['cpi'], transform=composed, T_channel=param['with_T'])
print("Data prepared. #Samples:{}.".format(len(dataset)))
print('Sample: {\'image\':',dataset[0]['image'].shape,'\'label\':',dataset[0]['label'],'}')

### Initialize the model
net = ConvMultiHypoNet(param['input_channel'], param['dim_out'], param['fc_input'], num_components=param['num_components'])
# net = ConvMixtureDensityNet(param['input_channel'], param['dim_out'], param['fc_input'], num_components=param['num_components'])
myNet = NetworkManager(net, loss_function_dict={}, verbose=False, device=param['device'])
myNet.build_Network()
myNet.model.load_state_dict(torch.load(model_path))
myNet.model.eval() # with BN layer, must run eval first

### Visualize
fig, ax = plt.subplots()
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')
for idx in idc:

    plt.cla()
    
    img    = dataset[idx]['image']
    traj   = np.array(dataset[idx]['traj'])
    hyposM = myNet.inference(img) # for WTA
    # alpha, mu, sigma = myNet.inference(img, mdn=True) # for MDN

    ### Kalman filter ###
    X0 = np.array([[traj[0,0], traj[0,1], 0, 0]]).transpose()
    kf_model = zfilter.model_CV(X0, Ts=1)
    P0 = zfilter.fill_diag((1,1,1,1))
    Q  = zfilter.fill_diag((0.1,0.1,0.1,0.1))
    R  = zfilter.fill_diag((0.1,0.1))
    KF = zfilter.KalmanFilter(kf_model, P0, Q, R)
    Y = [np.array(traj[1,:]), np.array(traj[2,:]), 
         np.array(traj[3,:]), np.array(traj[4,:])]
    for kf_i in range(len(Y) + dataset.T):
        if kf_i<len(Y):
            KF.one_step(np.array([[0]]), np.array(Y[kf_i]).reshape(2,1))
        else:
            KF.predict(np.array([[0]]), evolve_P=False)
            KF.append_state(KF.X)
    ### ------------- ###

    hypos_clusters = utils_test.fit_DBSCAN(hyposM[0], eps=0.5, min_sample=3) # DBSCAN
    mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters) # Gaussian fitting

    utils_test.plot_Gaussian_ellipses(ax, mu_list, std_list)
    # utils_test.plot_parzen_distribution(ax, hyposM[0,:,:])
    utils_test.plot_on_simmap(ax, dataset[idx], hyposM[0,:,:])

    # utils_test.plot_on_simmap(ax, dataset[idx], None)
    # utils_test.plot_mdn_output(ax, alpha, mu, sigma)

    for i, hc in enumerate(hypos_clusters):
        ax.scatter(hc[:,0], hc[:,1], marker='x', label=f"est{i+1}")

    plt.plot(KF.X[0], KF.X[1], 'mo', label='KF')
    ax.add_patch(patches.Ellipse(KF.X[:2], 3*KF.P[0,0], 3*KF.P[1,1], fc='g', zorder=0))

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

