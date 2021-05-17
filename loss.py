import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import sys
import scipy.spatial.distance
import math
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import open3d as o3d
import glob
import time
from torch_geometric.nn import knn_interpolate, fps
import json
from collections import Counter, OrderedDict

batch_size = 2
batch_size_test = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def gagcn_loss(pred,labels, n_cls, smoothing = True):
        
    if smoothing:
        labels = labels.view(-1,1).squeeze()
        labels = labels.contiguous().view(-1)
        pred = pred.permute(0, 2, 1).contiguous()
        pred = pred.view(-1, n_cls)#####################
        eps = 0.2
        n_class = pred.size(1)
        
        one_hot_tensor = torch.zeros_like(pred).scatter(1, labels.view(-1,1), 1)
        one_hot_tensor = one_hot_tensor * (1-eps) + (1-one_hot_tensor) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        
        loss = -(one_hot_tensor * log_prb).sum(dim=-1).mean()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
        loss1 = criterion(pred, labels)
        loss = loss1 
    
    return loss

def multitask_loss(output_pred,labels,n_cls,log_vars,smoothing=True):
    loss = 0
    criterion = nn.MSELoss()
    cnt = 0
    stds = (torch.exp(-log_vars[0])**(1/2)).to(device)
    precision1 = 1 / ( 1*(stds**2) )
    ls = gagcn_loss(output_pred[0],labels[0],n_cls)
    loss = precision1 * ls + torch.log(stds)

    stds = (torch.exp(-log_vars[1])**(1/2)).to(device)
    precision2 = 1 / ( 2*(stds**2) )
    ls = criterion(output_pred[1],labels[1])
    ls2 = precision2 * ls + torch.log(stds)
    loss += ls2

    return loss,ls2

def disc_loss(output,num, x=None,y=None,pred=False, smoothing=False):
    label = torch.tensor(batch_size)
    real_label = num
    label.resize_(batch_size).fill_(real_label)
    label = label.to(device)
    if pred == True:
        num = x.size(1)
        criterion = nn.MSELoss()
        loss = criterion(x,y)
    else:
        if smoothing:
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(output, label)        
        else:
            labels = label.view(-1,1).squeeze()
            labels = labels.contiguous().view(-1)
            pred = output.contiguous() 
            pred = pred.view(-1, 2)#####################
            eps = 0.2
            n_class = pred.size(1)

            one_hot_tensor = torch.zeros_like(pred).scatter(1, labels.view(-1,1), 1)
            one_hot_tensor = one_hot_tensor * (1-eps) + (1-one_hot_tensor) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot_tensor * log_prb).sum(dim=-1).mean()
        
    return loss