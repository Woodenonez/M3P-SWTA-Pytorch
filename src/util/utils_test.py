import os
import math

import torch
import numpy as np

from sklearn.cluster import DBSCAN

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from util import utils_data
from data_handle import sid_object

def kernel_Gaussian(X, Mu=[0,0], Sigma=[[0.05,0],[0,0.05]]):
    X = np.array(X).reshape(-1,1)
    Mu = np.array(Mu).reshape(-1,1)
    Sigma = np.array(Sigma)
    prob = ( 1/np.sqrt((2*math.pi)**2*np.linalg.det(Sigma))
           * np.exp(-(np.transpose(X-Mu).dot(np.linalg.inv(Sigma)).dot(X-Mu)) / 2) )
    return prob

def est_kernel_density(x, data, kernel): # Parzen window
    # x    = np.array([x1,x2])
    # data = np.array([[data1_x1,data1_x2], [data2_x1,data2_x2], ...])
    n = data.shape[0]
    h = 1 # bandwidth
    f = 1/(n*h) * sum([kernel((x-d)/h) for d in data])
    return f

def plot_parzen_distribution(ax, data):
    # data = np.array([[data1_x1,data1_x2], [data2_x1,data2_x2], ...])
    x = np.arange(np.min(data[:,0])-1, np.max(data[:,0])+1, step=0.05)
    y = np.arange(np.min(data[:,1])-1, np.max(data[:,1])+1, step=0.05)
    xx, yy = np.meshgrid(x, y)
    xy = np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1)), axis=1).astype(np.float32)
    f = np.array([])
    for i in range(xy.shape[0]):
        f = np.append(f, est_kernel_density(xy[i,:], data, kernel=kernel_Gaussian))
    ff = f.reshape(xx.shape)

    ax.contourf(xx, yy, ff, cmap='Greys')

def cal_GauProb(mu, sigma, x):
    '''
    Description:
        Return the probability of "data" given MoG parameters "mu" and "sigma".
    Arguments:
        mu    (BxGxC) - The means of the Gaussians. 
        sigma (BxGxC) - The standard deviation of the Gaussians.
        x     (BxC)   - A batch of data points.
    Return:
        prob (BxG) - The probability of each point in the distribution in the corresponding mu/sigma index.
    '''
    x = x.unsqueeze(1).expand_as(mu) # BxC -> Bx1xC -> BxGxC
    prob = torch.rsqrt(torch.tensor(2*math.pi)) * torch.exp(-((x - mu) / sigma)**2 / 2) / sigma
    return torch.prod(prob, dim=2) # overall probability for all output's dimensions in each component, BxG

def cal_multiGauProb(alp, mu, sigma, x):
    '''
    Description:
        Return the probability of "data" given MoG parameters "mu" and "sigma".
    Arguments:
        alp   (BxG)   - Component's weight.
        mu    (BxGxC) - The means of the Gaussians. 
        sigma (BxGxC) - The standard deviation of the Gaussians.
        x     (BxC)   - A batch of data points.
    Return:
        prob (Bx1) - The probability of each point in the distribution in the corresponding mu/sigma index.
    '''
    prob = alp * cal_GauProb(mu, sigma, x) # BxG
    prob = torch.sum(prob, dim=1) # Bx1, overall prob for each batch (sum is for all compos)
    return prob

def cal_multiGauProbDistr(xx, yy, alp, mu, sigma):
    xy = np.concatenate((xx.reshape(-1,1), yy.reshape(-1,1)), axis=1).astype(np.float32)
    p = np.array([])
    for i in range(xy.shape[0]):
        p = np.append( p, cal_multiGauProb(alp, mu, sigma, x=torch.tensor(xy[i,:][np.newaxis,:])).detach().numpy() )
    p[np.where(p<max(p)/10)] = 0
    return p.reshape(xx.shape)

def sigma_limit(mu, sigma, nsigma=3):
    # nsigma: 1 -> 0.6827   2 -> 0.9545   3 -> 0.9974
    x_scope = [(mu-nsigma*sigma)[0,:,0], (mu+nsigma*sigma)[0,:,0]]
    y_scope = [(mu-nsigma*sigma)[0,:,1], (mu+nsigma*sigma)[0,:,1]]
    x_min = torch.min(x_scope[0])
    x_max = torch.max(x_scope[1])
    y_min = torch.min(y_scope[0])
    y_max = torch.max(y_scope[1])
    if x_min !=  torch.min(abs(x_scope[0])):
        x_min = -torch.min(abs(x_scope[0]))
    if x_max !=  torch.max(abs(x_scope[1])):
        x_max = -torch.max(abs(x_scope[1]))
    if y_min !=  torch.min(abs(y_scope[0])):
        y_min = -torch.min(abs(y_scope[0]))
    if y_max !=  torch.max(abs(y_scope[1])):
        y_max = -torch.max(abs(y_scope[1]))
    return [x_min, x_max], [y_min, y_max]

def draw_probDistribution(ax, alp, mu, sigma, nsigma=3, step=0.5, colorbar=False, toplot=True):
    '''
    Arguments:
        ax            - Axis
        alp   (BxG)   - (alpha) Component's weight.
        mu    (BxGxC) - The means of the Gaussians. 
        sigma (BxGxC) - The standard deviation of the Gaussians.
    '''
    # ================= Register Colormap ================START
    ncolors = 256
    color_array = plt.get_cmap('gist_rainbow')(range(ncolors)) # get colormap
    color_array[:,-1] = np.linspace(0.5,1,ncolors) # change alpha values
    color_array[:,-1][:25] = 0
    map_object = matplotlib.colors.LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array) # create a colormap object
    plt.register_cmap(cmap=map_object) # register this new colormap with matplotlib
    # ================= Register Colormap ==================END

    xlim, ylim = sigma_limit(mu, sigma, nsigma=nsigma)
    x = np.arange(xlim[0].detach().numpy(), xlim[1].detach().numpy(), step=step)
    y = np.arange(ylim[0].detach().numpy(), ylim[1].detach().numpy(), step=step)
    xx, yy = np.meshgrid(x, y)

    pp = cal_multiGauProbDistr(xx, yy, alp, mu, sigma)

    if toplot:
        cntr = ax.contourf(xx, yy, pp, cmap="rainbow_alpha")
        if colorbar:
            plt.colorbar(cntr, ax=ax)

    return xx,yy,pp

def fit_DBSCAN(data, eps, min_sample):
    # data = np.array([[],[],[]])
    clustering = DBSCAN(eps=eps, min_samples=min_sample).fit(data)
    nclusters = len(list(set(clustering.labels_)))
    if -1 in clustering.labels_:
        nclusters -= 1
    clusters = []
    for i in range(nclusters):
        cluster = data[clustering.labels_==i,:]
        clusters.append(cluster)
    return clusters

def fit_cluster2gaussian(clusters):
    mu_list  = []
    std_list = []
    for cluster in clusters:
        mu_list.append(np.mean(cluster, axis=0))
        std_list.append(np.std(cluster, axis=0))
    return mu_list, std_list

def plot_Gaussian_ellipses(ax, mu_list, std_list, alpha=None, label=None):
    for mu, std in zip(mu_list, std_list):
        patch = patches.Ellipse(mu, std[0], std[1], fc='y', ec='purple', alpha=alpha, label=label)
        ax.add_patch(patch)

def plot_mdn_output(ax, alpha, mu, sigma):
    ax.plot(mu[0,0], mu[0,1], 'ro', label='est')
    patch = patches.Ellipse(mu[0,:], sigma[0,0], sigma[0,1], fc='y')
    ax.add_patch(patch)
    for i in range(len(alpha)-1):
        if alpha[i+1] > 0:
            ax.plot(mu[i+1,0], mu[i+1,1], 'ro')
            patch = patches.Ellipse(mu[i+1,:], sigma[i+1,0], sigma[i+1,1], fc='y')
            ax.add_patch(patch)

def plot_on_simmap(ax, sample, hyposM, label_scale=1, plot_other=True):
    label = sample['label'] * label_scale
    traj  = np.array(sample['traj'])
    scene_index = sample['index']
    time_step = sample['time']

    map_idx, path_idx, interact = utils_data.index2map(scene_index) # map parameters
    stagger, vmax, target_size, ts = (0.2, 1, 0.5, 0.2)   # object parameters

    graph = sid_object.Graph(map_idx, block=False)
    graph.plot_map(ax, clean=1)
    if interact & plot_other:
        obs_pos = graph.get_obs_path(ts=ts, start=10-time_step*ts)
        ax.plot(obs_pos[0][0], obs_pos[0][1], 'r^', label='other')

    ax.plot(traj[-1,0], traj[-1,1], 'ko', label='target')
    ax.plot(traj[:-1,0], traj[:-1,1], 'k.') # past
    ax.plot(label[0], label[1], 'bo', label="GT")
    if hyposM is not None:
        ax.scatter(hyposM[:,0], hyposM[:,1], c='r', marker='.', label="est")

def plot_on_realmap(): pass