import os, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

from net_module.net import ConvMultiHypoNet
from data_handle import data_handler as dh
from data_handle import dataset as ds

from net_module import loss_functions as metrics

from util import utils_test
import pre_load

print("Program: evaluation\n")

### Config
config_file = 'sdd_15_swta_test.yml'

data_from_zip = False
composed = torchvision.transforms.Compose([dh.ToTensor()])
Dataset = ds.ImageStackDataset
Net = ConvMultiHypoNet

### Prepare
root_dir = Path(__file__).resolve().parents[1]
dataset, datahandler, net = pre_load.main_test_pre(root_dir, config_file, Dataset, data_from_zip, composed, Net)

### Evaluation
idx_start, idx_end = 0, len(dataset)-1
fig, ax = plt.subplots()
idc = np.linspace(idx_start,idx_end,num=idx_end-idx_start).astype('int')

lossWMD_list    = []
lossminMD_list  = []
lossNLL_list    = []
lossOracle_list = []
for idx in idc:
    print(f'\r{idx}/{idx_end}  ', end='')
    
    _, label, _, _, hyposM, _ = pre_load.main_test(dataset, net, idx=idx)

    hypos_clusters = utils_test.fit_DBSCAN(hyposM[0], eps=50, min_sample=3) # DBSCAN
    mu_list, std_list = utils_test.fit_cluster2gaussian(hypos_clusters) # Gaussian fitting
    alp = torch.ones((len(mu_list))).unsqueeze(0)
    mu  = torch.tensor(np.array(mu_list)).unsqueeze(0)
    std = torch.tensor(np.array(std_list)).unsqueeze(0)

    lossOracle = metrics.loss_CentralOracle(mu, label)
    lossOracle_list.append(lossOracle[0].detach().float().item())

    lossNLL = metrics.loss_NLL(alp, mu, std, label)
    lossNLL_list.append(lossNLL.detach().float().item())

    lossMD, lossWMD = metrics.loss_MaDist(alp, mu, std, label)
    lossWMD_list.append(lossWMD[0].detach().float().item())
    lossminMD_list.append(torch.min(lossMD).detach().float().item())

print()

print(f'Config. file: {config_file}; Avg Oracle loss: {sum(lossOracle_list)/len(lossOracle_list)},',
                                   f'Avg minMD loss: {sum(lossminMD_list)/len(lossminMD_list)},',
                                   f'Avg NLL loss: {sum(lossNLL_list)/len(lossNLL_list)},',
                                   f'Avg WMD loss: {sum(lossWMD_list)/len(lossWMD_list)},',)


