import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import random
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

def one_hot(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels] 

def calculate_shape_IoU(pred_np, seg_np,n_cls):
    seg_num = n_cls #######################
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        parts = range(seg_num) #0,1
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious

def read_json(file):
    f = open(file,)
    x = json.load(f)
    return x

def sampling(pts, pts_cat):    
    res1 = np.concatenate((pts,np.reshape(pts_cat, (pts_cat.shape[0], 1))), axis= 1)
    res = np.asarray(random.choices(res1, weights=None, cum_weights=None, k=2048))
    images = res[:, 0:3]
    categories = res[:, 3]
    categories-=np.ones(categories.shape)
    return images, categories    
######################################
#AUGMENTATION
class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.001, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud

class Translate_pointcloud(object):
    def __call__(self,pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud

class Jitter_pointcloud(object): 
    def __call__(self,pointcloud, sigma=0.01, clip=0.02):
        self.pointcloud = pointcloud
        self.N, self.C = self.pointcloud.shape
        self.sigma = sigma
        self.pointcloud += np.clip(self.sigma * np.random.randn(self.N,self.C), -1*clip,clip)
        return self.pointcloud

class Rotate_pointcloud(object):
    def __call__(self,pointcloud):
        self.pointcloud = pointcloud
        self.theta = np.pi*2 * np.random.uniform()
        self.rotation_matrix = np.array([[np.cos(self.theta), -np.sin(self.theta)],[np.sin(self.theta), np.cos(self.theta)]])
        self.pointcloud[:,[0,2]] = self.pointcloud[:,[0,2]].dot(self.rotation_matrix) # random rotation (x,z)
        return self.pointcloud
        return images, categories
    
def read_pts(file):
    verts = np.genfromtxt(file)
    return verts

def read_seg(file):
    verts = np.genfromtxt(file, dtype= (int))
    return verts

####################################
#CUSTOM DATASET
class CustomDataset(Dataset):
    def __init__(self,path,data,transform,valid=False):
        self.path = path
        self.transform = transform
        self.valid = valid
        self.data = data
        self.files = []
        new_dir = self.path + r'/points'

        if self.valid == True:
            for i in self.data:      
                sample = {}
                fname1 = i.split('/')[-1]
                fname = new_dir+'/'+fname1+'.pts'
                label = self.path + '/points_label/' + fname1+'.seg'
                with open(fname,'r') as f:
                    x1 = read_pts(f)
                with open(label,'r') as f:
                    y1 = read_seg(f)
                x1,y1 = sampling(x1,y1)
                sample['path'] = x1
                sample['labels'] = y1
                self.files.append(sample)
                
        else:        
            for i in self.data:
                fname = i.split('/')[-1]
                sample = {}
                sample['path'] = new_dir+'/'+fname+'.pts'
                sample['labels'] = self.path + '/points_label/' + fname+'.seg'
                self.files.append(sample)

        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        if self.valid == False:
            x = self.files[idx]['path']
            y = self.files[idx]['labels']
            with open(x,'r') as f:
                x1 = read_pts(f)
            with open(y,'r') as f:
                y1 = read_seg(f)

            x_sample,y_sample = sampling(x1,y1)
        else:
            x_sample = self.files[idx]['path']
            y_sample = self.files[idx]['labels']
        
        
        #shuffling points
        
        if self.valid == False:
            indices = list(range(x_sample.shape[0]))
            np.random.shuffle(indices)
            x_sample = x_sample[indices]
            y_sample = y_sample[indices]
        
        if self.transform != None:
            x_sample = self.transform(x_sample)
            
        return {"points": np.array(x_sample, dtype='float32'), "labels": y_sample.astype(int)}

train_transforms = transforms.Compose([
                    #Jitter_pointcloud(),
                    #Rotate_pointcloud(),
                    #Translate_pointcloud(),
                    #RandRotation_z(),
                    #RandomNoise()
                    ])