import os, sys
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

# 1. Architecture
from net_module.net import ConvMultiHypoNet, ConvMixtureDensityNet
# 2. Training manager
from network_manager import NetworkManager
# 3. Data handler
from data_handle import data_handler_zip as dh

from net_module import loss_functions as metrics
from net_module import module_mdn

from util import utils_test
from util import utils_yaml

print("Program: animation\n")

### Config file name
# config_file = 'mdn_20_test.yml'
config_file = 'ewta_20_test.yml'

### Load parameters and define paths
root_dir = Path(__file__).resolve().parents[1]
param_path = os.path.join(root_dir, 'Config/', config_file)
param = utils_yaml.from_yaml(param_path)
param['device'] = 'cuda'

model_path = os.path.join(root_dir, param['model_path'])

zip_path  = os.path.join(root_dir, param['zip_path'])
csv_path  = os.path.join(param['data_name'], param['label_csv'])
data_dir  = param['data_name']
print("Model:", model_path)

### Prepare data
composed = torchvision.transforms.Compose([dh.ToTensor()])
dataset = dh.ImageStackDataset(zip_path, csv_path, data_dir, channel_per_image=param['cpi'], transform=composed, T_channel=param['with_T'])
print("Data prepared. #Samples:{}.".format(len(dataset)))
print('Sample: {\'image\':',dataset[0]['image'].shape,'\'label\':',dataset[0]['label'],'}')

# fig, axes = plt.subplots(5,5)
### Initialize the model
net = ConvMultiHypoNet(param['input_channel'], param['dim_out'], param['fc_input'], num_components=param['num_components'])
# net = ConvMixtureDensityNet(param['input_channel'], param['dim_out'], param['fc_input'], num_components=param['num_components'])
myNet = NetworkManager(net, loss_function_dict={}, device=param['device'], verbose=False)
myNet.build_Network()
myNet.model.load_state_dict(torch.load(model_path))
myNet.model.eval() # with BN layer, must run eval first

### Visualize
idx_start, idx_end = 0, 1000 #len(dataset)-1
fig, ax = plt.subplots()
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')
lossWMD_list = []
lossminMD_list = []
lossNLL_list = []
lossOracle_list = []
runtime_list = []
for idx in idc:
    print(f'\r{idx}/{idx_end}  ', end='')

    plt.cla()

    start = perf_counter()
    
    img    = dataset[idx]['image']
    label  = dataset[idx]['label']
    traj   = np.array(dataset[idx]['traj'])
    hyposM = myNet.inference(img)

    # alp, mu, std = myNet.inference(img, mdn=True)
    # alp = torch.tensor(alp)
    # mu  = torch.tensor(mu)
    # std = torch.tensor(std)
    # alp, mu, std = module_mdn.take_goodCompo(alp, mu, std, 0.1)

    hypos_clusters = utils_test.fit_DBSCAN(hyposM[0], eps=50, min_sample=3) # DBSCAN
    mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters) # Gaussian fitting
    alp = torch.ones((len(mu_list),)) / len(mu_list)
    mu  = torch.tensor(np.array(mu_list))
    std = torch.tensor(np.array(std_list))

    runtime_list.append(perf_counter()-start)

    lossOracle = metrics.loss_CentralOracle(mu, label)
    lossOracle_list.append(lossOracle.detach().float().item())

    lossNLL = metrics.loss_NLL(alp.unsqueeze(0), mu.unsqueeze(0), std.unsqueeze(0), label.unsqueeze(0))
    lossNLL_list.append(lossNLL.detach().float().item())

    lossMD_list, lossWMD = metrics.loss_MaDist(alp, mu, std, label)
    lossWMD_list.append(lossWMD.detach().float().item())
    lossminMD_list.append(torch.min(lossMD_list).detach().float().item())

print()

print(f'Config. file: {config_file}; Avg Oracle loss: {sum(lossOracle_list)/len(lossOracle_list)},',
                                   f'Avg minMD loss: {sum(lossminMD_list)/len(lossminMD_list)},',
                                   f'Avg NLL loss: {sum(lossNLL_list)/len(lossNLL_list)},',
                                   f'Avg WMD loss: {sum(lossWMD_list)/len(lossWMD_list)},',)
                                #    f'Avg runtime: {sum(runtime_list)/len(runtime_list)}s')
# h = ax.hist(np.array(lossWMD_list), bins=20, alpha=0)
# plt.plot(h[1][:-1]+(h[1][-1]-h[1][-2])/2, h[0],'bx--',label='ppp')
# plt.show()

