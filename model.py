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

def knn(x, k):

    x1 = x.permute(0,2,1)
    idx = torch.cdist(x1,x1,p=2).topk(k=k,dim=-1,largest=False)[1]
    return idx


def knn_dilate(x, k, rate, dilation_type):
    batch_size = x.size(0)
    device = torch.device('cuda')
    k2 = rate*k

    x1 = x.permute(0,2,1)
    idx = torch.cdist(x1,x1,p=2).topk(k=k2,dim=-1,largest=False)[1]
    idx = idx.cpu().detach().numpy()
        
    if dilation_type == 'order':
    #Ordered sampling
        idx = idx[:, :, ::rate]
    else:
    #Random sampling
        idx_i = np.random.choice(idx.shape[2],k,replace=False)
        idx = idx[:, :, idx_i]
    idx = torch.tensor(idx).to(device)
    return idx

def get_graph_feature(x, pos, k, idx=None, dim9=False, dilation=True, rate = 2,dilation_type='order'):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    x_pos = pos.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            if dilation == True:
                idx = knn_dilate(pos, k=k, rate= rate, dilation_type = dilation_type)
            else:
                idx = knn(pos, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.contiguous().view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    x_pos = x_pos.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature_pos = x_pos.view(batch_size*num_points, -1)[idx, :]
    z = feature_pos.size(1)
    feature_pos = feature_pos.view(batch_size, num_points, k, z) 
    x_pos = x_pos.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1)
    
    pd = nn.PairwiseDistance(p=2,keepdim=True)
    x_pos2 = x_pos.permute(0,3,2,1)
    feature_pos2 = feature_pos.permute(0,3,2,1)
    feature_add =pd(x_pos2,feature_pos2)
    feature_add = feature_add.view(batch_size,num_points,k,1)

    feature = torch.cat((feature-x, feature_pos-x_pos, feature_add, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature      # (batch_size, 3*num_dims+1, num_points, k)

##################################################
#ATTENTION MODULES
class eca_layer(nn.Module):
    """
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.softmax = nn.Softmax(dim=1)#nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, pts = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x) #b,c,1

        # Two different branches of ECA module
        y = y.permute(0,2,1) #bs*1*c
        y = self.conv(y)
        y = y.permute(0,2,1)

        # Multi-scale information fusion
        y = self.softmax(y)

        return x * y.expand_as(x)
    
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1)
        return x * y.expand_as(x)
#######################################################
#SPATIAL TRANSFORMER NETWORK
class Transform_Net(nn.Module):
    def __init__(self,k,norm='batch'):#, args):
        super(Transform_Net, self).__init__()
        #self.args = args
        self.norm=norm
        self.k = k
        if self.norm == 'batch':
            self.bn1 = nn.BatchNorm2d(64)#nn.GroupNorm(32,self.x)
            self.bn1_2 = nn.BatchNorm2d(64)#nn.GroupNorm(32,out_dims)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)
        else:
            self.bn1 = nn.GroupNorm(32,64)
            self.bn1_2 = nn.GroupNorm(32,64)
            self.bn2 = nn.GroupNorm(32,128)
            self.bn3 = nn.GroupNorm(32,1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)
        
        self.conv1 = nn.Sequential(nn.Conv2d(self.k*3+1, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv1_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                    self.bn1_2,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 256, bias=False)

        self.transform = nn.Linear(256, k*k)#3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(k, k))#3, 3))

    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1_2(x)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        m = nn.AdaptiveMaxPool2d((x.size(2),1))#.view(batch_size, -1)   
        x = m(x).squeeze(-1)
        #x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        m = nn.AdaptiveMaxPool1d(1)#.view(batch_size, -1)   
        x = m(x).squeeze(-1)
        #x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn4(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn5(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)
        
        #initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(batch_size,1,1)
        if x.is_cuda:
            init=init.cuda()
        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, self.k,self.k) + init            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x
###############################################

class GraphConv(nn.Module):
    def __init__(self,k, in_dims, mid_dims=64, out_dims=None, layer=2,norm = 'batch', pyramid = False, dilation = 'order', dilation_rate=1, bias = True):
        super(GraphConv, self).__init__()
        self.dilation_rate = dilation_rate
        self.bias = bias
        self.dilation = dilation
        self.k = k
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.x = mid_dims #64
        if self.out_dims == None:
            self.out_dims = self.x
        self.norm = norm
        self.layer = layer
        self.pyramid = pyramid
        
        if self.layer == 2:
            if self.norm == 'batch':
                self.bn1 = nn.BatchNorm2d(self.x)#nn.GroupNorm(32,self.x)
                self.bn2 = nn.BatchNorm2d(self.out_dims)#nn.GroupNorm(32,out_dims)
            else:
                self.bn1 = nn.GroupNorm(32,self.x)
                self.bn2 = nn.GroupNorm(32,out_dims)

            self.conv1 = nn.Sequential(nn.Conv2d(self.in_dims, self.x, kernel_size=1, bias=self.bias),
                                       self.bn1,
                                       nn.LeakyReLU(negative_slope=0.2))
            self.conv2 = nn.Sequential(nn.Conv2d(self.x, self.out_dims, kernel_size=1, bias=self.bias),
                                       self.bn2,
                                       nn.LeakyReLU(negative_slope=0.2))
            if self.pyramid == True:
                if self.norm == 'batch':
                    self.bn1_2 = nn.BatchNorm2d(self.x)#nn.GroupNorm(32,self.x)
                    self.bn2_2 = nn.BatchNorm2d(self.out_dims)#nn.GroupNorm(32,out_dims)
                    self.bn1_3 = nn.BatchNorm2d(self.x)#nn.GroupNorm(32, self.x)
                    self.bn2_3 = nn.BatchNorm2d(self.out_dims)#nn.GroupNorm(32, self.out_dims)
                    self.bn_add = nn.BatchNorm2d(self.out_dims)#nn.GroupNorm(32, self.out_dims)
                else:
                    self.bn1_2 = nn.GroupNorm(32,self.x)
                    self.bn2_2 = nn.GroupNorm(32,out_dims)
                    self.bn1_3 = nn.GroupNorm(32, self.x)
                    self.bn2_3 = nn.GroupNorm(32, self.out_dims)
                    self.bn_add = nn.GroupNorm(32, self.out_dims)

                self.conv1_2 = nn.Sequential(nn.Conv2d(self.in_dims, self.x, kernel_size=1, bias=self.bias),
                           self.bn1_2,
                           nn.LeakyReLU(negative_slope=0.2))
                self.conv2_2 = nn.Sequential(nn.Conv2d(self.x, out_dims, kernel_size=1, bias=self.bias),
                                           self.bn2_2,
                                           nn.LeakyReLU(negative_slope=0.2))
                self.conv1_3 = nn.Sequential(nn.Conv2d(self.in_dims, self.x, kernel_size=1, bias=self.bias),
                                            self.bn1_3,
                                            nn.LeakyReLU(negative_slope=0.2))
                self.conv2_3 = nn.Sequential(nn.Conv2d(self.x, self.out_dims, kernel_size=1, bias=self.bias),
                                            self.bn2_3,
                                            nn.LeakyReLU(negative_slope=0.2))

                self.conv_add = nn.Sequential(nn.Conv1d(self.out_dims*3, self.out_dims, kernel_size=1, bias=self.bias),
                                        self.bn_add,
                                        nn.LeakyReLU(negative_slope=0.2))

        elif self.layer==1:
            if self.norm == 'batch':
                #pass
                self.bn1 = nn.BatchNorm2d(self.out_dims)
            else:
                self.bn1 = nn.GroupNorm(32,self.out_dims)#nn.BatchNorm2d(self.out_dims)

            self.conv1 = nn.Sequential(nn.Conv2d(self.in_dims, self.out_dims, kernel_size=1, bias=self.bias),
                                      self.bn1,
                                      nn.LeakyReLU(negative_slope=0.2))

            if self.pyramid == True:
                if self.norm == 'batch':
                    #pass
                    self.bn1_3 = nn.BatchNorm2d(self.out_dims)
                    self.bn1_2 = nn.BatchNorm2d(self.out_dims)
                else:
                    self.bn1_3 = nn.GroupNorm(32, self.out_dims)
                    self.bn1_2 = nn.GroupNorm(32,self.out_dims)
                
                self.conv1_2 = nn.Sequential(nn.Conv2d(self.in_dims, self.out_dims, kernel_size=1, bias=self.bias),
                                            self.bn1_2,
                                            nn.LeakyReLU(negative_slope=0.2))
                
    def forward(self, x, pos):
        if self.layer == 2:
            if self.pyramid == False:
                if self.dilation_rate != 1:
                    x1 = get_graph_feature(x, pos, k=self.k, dilation=False, dilation_type = self.dilation)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
                else:
                    x1 = get_graph_feature(x, pos, k=self.k, dilation=True, dilation_type = self.dilation)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
                x1 = self.conv1(x1)
                x1 = self.conv2(x1)
                x1_max = x1.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
                x1 = x1_max#self.se1(x1_max)
                residual = x1
            
            else:
                x1 = get_graph_feature(x, pos, k=self.k, dilation=True, rate=2, dilation_type = self.dilation)
                x1 = self.conv1(x1)
                x1 = self.conv2(x1)
                x1 = x1.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
                
                x2 = get_graph_feature(x, pos, k=self.k, dilation=True, rate=4, dilation_type = self.dilation)
                x2 = self.conv1_2(x2)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
                x2 = self.conv2_2(x2)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
                x2 = x2.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
                
                x3 = get_graph_feature(x, pos, k=self.k, dilation=True, rate=6, dilation_type = self.dilation)
                x3 = self.conv1_3(x3)
                x3 = self.conv2_3(x3)
                x1_3 = x3.max(dim=-1, keepdim=False)[0]
                x1 = torch.cat((x1,x2,x1_3),1)
                x1 = self.conv_add(x1)
                residual = x1
                
        elif self.layer == 1:
            if self.pyramid == False:
                if self.dilation_rate != 1:
                    x1 = get_graph_feature(x, pos, k=self.k, dilation=True, rate=self.dilation_rate, dilation_type = self.dilation)
                else:
                    x1 = get_graph_feature(x, pos, k=self.k, dilation=False, rate=1, dilation_type = self.dilation)
                x1 = self.conv1(x1)
                m = nn.AdaptiveMaxPool2d((x1.size(2),1))#.view(batch_size, -1)   
                x1_max = m(x1).squeeze(-1)

                residual = x1_max
            else:
                x1 = get_graph_feature(x, pos, k=self.k, dilation=True, rate=self.dilation_rate, dilation_type = self.dilation)
                x1 = self.conv1(x1)
                x1 = x1.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
                
                x2 = get_graph_feature(x, pos, k=self.k, dilation=True, rate=self.dilation_rate*2, dilation_type = self.dilation)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
                x2 = self.conv1_2(x2)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
                x1_2 = x2.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
                
                x1 = x1+x1_2#+x1_3

                residual = x1

        return residual
    
######################################################
#GAGCN
class GAGCN(nn.Module):
    def __init__(self, seg_num_all, k, pyramid, dilation, dilation_rate, norm = 'group', bias = True):
        super(GAGCN, self).__init__()
        #self.args = args
        self.dilation_rate = dilation_rate
        self.bias = bias
        self.norm = norm
        self.seg_num_all = seg_num_all
        self.k = k
        self.device = torch.device('cuda')
        self.pyramid = pyramid
        self.dilation = dilation
        self.x = 64
        self.device = torch.device('cuda')
        self.transform_net = Transform_Net(3,norm=self.norm)
        self.edge1 = GraphConv(k=self.k, in_dims=9+1, out_dims= 128, layer=1, norm=self.norm, pyramid = self.pyramid[0], dilation = self.dilation, dilation_rate = self.dilation_rate[0],bias = self.bias)
        self.edge2 = GraphConv(k=self.k, in_dims=self.x*2*2+3+1, out_dims= 256, layer=1, norm=self.norm, pyramid = self.pyramid[1], dilation = self.dilation, dilation_rate = self.dilation_rate[1],bias = self.bias)
        self.edge3 = GraphConv(k=self.k, in_dims=self.x*2*2*2+3+1, out_dims= 1024, layer=1, norm=self.norm, pyramid = self.pyramid[2], dilation = self.dilation, dilation_rate = self.dilation_rate[2],bias = self.bias)
        self.edge4 = GraphConv(k=self.k, in_dims=3+2*(256+1024)+1, out_dims= 256, layer=1, norm=self.norm, pyramid = self.pyramid[3], dilation = self.dilation, dilation_rate = self.dilation_rate[3],bias = self.bias) #bottleneck
        self.edge5 = GraphConv(k=self.k, in_dims=3+2*(128+256)+1, out_dims= 256, layer=1, norm=self.norm, pyramid = self.pyramid[4], dilation = self.dilation, dilation_rate = self.dilation_rate[4],bias = self.bias)
        self.edge6 = GraphConv(k=self.k, in_dims=3+2*(256)+1, out_dims= 256, layer=1, norm=self.norm, pyramid = self.pyramid[4], dilation = self.dilation, dilation_rate = self.dilation_rate[5],bias = self.bias)              
        
        if self.norm == 'batch':
            #pass
            self.bn4 = nn.BatchNorm1d(256)
            self.bn5 = nn.BatchNorm1d(256)
        elif self.norm == 'group':
            self.bn4 = nn.GroupNorm(32, 256)
            self.bn5 = nn.GroupNorm(32, 256)
            
        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=self.bias),
                                  self.bn4,
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=self.bias),
                                  self.bn5,
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Conv1d(256, self.seg_num_all, kernel_size=1, bias=self.bias)

        
        self.dp3 = nn.Dropout(p=0.5)        
        
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        x0 = get_graph_feature(x, x, k=self.k, dilation=False, dilation_type = self.dilation)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)              # (batch_size, 3, 3)
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        
        pos = x
        pos_1 = pos
        x1 = self.edge1(x,pos_1)
        x1_upsample = x1
        x1_cpy = x1
        
        bs = torch.arange(x1.size(0)).to(self.device)
        bs = bs.repeat_interleave(x1.size(2))
        pos_upsample = pos.reshape(3,-1)
        pos = pos.permute(2,1,0).contiguous()
        pos = pos.view(pos.size(0)*pos.size(2),-1) #(n_pts*bs),channel
        x1 = x1.permute(2,1,0).contiguous()#n_pts,channel,bs
        x1 = x1.view(x1.size(0)*x1.size(2),-1) #(n_pts*bs),channel

        idx = fps(pos, batch=bs, ratio=0.375) #2048->768
        pos2 = pos[idx]
        pos2 = pos2.view(-1,pos2.size(1),batch_size)
        pos2 = pos2.permute(2,1,0) #batchsize,channel,pts
        pos2_1 = pos2
        x2 = x1[idx] #pts*bs,channel
        x2 = x2.view(-1,x1.size(1),batch_size) #pts,channel,bs
        x2 = x2.permute(2,1,0)#.view(x2.size(2),x2.size(1),x2.size(0))
        #########################
        x2 = self.edge2(x2,pos2_1)
        x2_upsample = x2
        
        bs2 = torch.arange(x2.size(0)).to(self.device)
        bs2 = bs2.repeat_interleave(x2.size(2))
        pos2_upsample = pos2.reshape(3,-1)
        pos2 = pos2.permute(2,1,0).contiguous()
        pos2 = pos2.view(pos2.size(0)*pos2.size(2),-1) #(n_pts*bs),channel
        x2 = x2.permute(2,1,0).contiguous()#view(x1.size(2),x1.size(1),x1.size(0))
        x2 = x2.view(x2.size(0)*x2.size(2),-1) #(n_pts*bs),channel

        idx = fps(pos2, batch=bs2, ratio=1/3) #768 -> 384
        pos3 = pos2[idx]
        pos3 = pos3.view(-1,pos3.size(1),batch_size)
        pos3 = pos3.permute(2,1,0) #batchsize,channel,pts 
        pos3_1 = pos3
        pos3_upsample = pos3.reshape(3,-1)
        x3 = x2[idx]
        x3 = x3.view(-1,x3.size(1),batch_size) #pts,channel,bs
        x3 = x3.permute(2,1,0)#.view(x3.size(2),x3.size(1),x3.size(0))
        ##########################
        #pos3 = pos2[:,idx]
        x3 = self.edge3(x3,pos3_1) #bottleneck
        x3_upsample = x3 #bs,channel,n_pts
        bs3 = torch.arange(x3.size(0)).to(self.device)
        bs3 = bs3.repeat_interleave(x3.size(2))
        pos3 = pos3_upsample
        pos2 = pos2_upsample
        pos = pos_upsample
        
        x3_2 = x3.reshape(x3.size(0)*x3.size(2),-1)
        x4 = knn_interpolate(x3_2,pos3.reshape(pos3.size(1),pos3.size(0)),pos2.reshape(pos2.size(1),pos2.size(0)),batch_x = bs3,batch_y = bs2)
        x4 = x4.reshape(batch_size,x3_2.size(1),-1)
        x4 = torch.cat((x4,x2_upsample),1)
        x4 = self.edge4(x4,pos2_1)
        
        x4_2 = x4.reshape(x4.size(0)*x4.size(2),-1)
        x5 = knn_interpolate(x4_2,pos2.reshape(pos2.size(1),pos2.size(0)),pos.reshape(pos.size(1),pos.size(0)),batch_x = bs2,batch_y = bs)
        x5 = x5.reshape(batch_size,x4_2.size(1),-1)
        x5 = torch.cat((x5,x1_upsample),1)
        x5 = self.edge5(x5,pos_1)
    
        x5 = self.edge6(x5,pos_1)

        residual = x5
               
        x = self.conv4(x5)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        
        x = self.conv5(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)

        x = self.dp3(x)
        x = self.conv6(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x
######################################################
#DISCRIMINATOR

class Discriminator(nn.Module):
    def __init__(self, k, dilation, dilation_rate, n_cls,out_dims, pyramid,norm='group', bias=True):
        super(Discriminator, self).__init__()
        self.pyramid = pyramid
        self.n_cls = n_cls
        self.bias = bias
        self.x = 64
        self.out_dims = out_dims
        self.device = torch.device('cuda')
        self.k = k
        self.dilation = dilation
        self.dilation_rate = dilation_rate
        self.norm = norm 
        if self.norm == 'batch':
            self.bn2 = nn.BatchNorm1d(self.out_dims[1])
            self.bn3 = nn.BatchNorm1d(self.out_dims[2])
            self.bn6 = nn.BatchNorm1d(128)
            self.bn7 = nn.BatchNorm1d(128)
        else:
            self.bn2 = nn.GroupNorm(32,self.out_dims[1])
            self.bn3 = nn.GroupNorm(32,self.out_dims[2])
            self.bn6 = nn.GroupNorm(32,256)
            self.bn7 = nn.GroupNorm(32,256)
            
        self.edge1 = GraphConv(k=self.k, in_dims=(n_cls+3)*2+3+1, mid_dims = self.out_dims[0],out_dims= self.out_dims[0], layer=2, norm=self.norm,pyramid = self.pyramid[0], dilation = self.dilation, dilation_rate = self.dilation_rate[0],bias = self.bias)
        
        self.edge2 = GraphConv(k=self.k, in_dims=self.out_dims[0]*2+3+1, mid_dims = self.out_dims[1],out_dims= self.out_dims[1], layer=2, norm=self.norm, pyramid = self.pyramid[1], dilation = self.dilation, dilation_rate = self.dilation_rate[1],bias = self.bias)
        
        self.edge3 = GraphConv(k=self.k, in_dims=self.out_dims[1]*2+3+1, mid_dims = self.out_dims[2], out_dims= self.out_dims[2], layer=2, norm=self.norm, pyramid = self.pyramid[2], dilation = self.dilation, dilation_rate = self.dilation_rate[2],bias = self.bias)
        self.edge3_2 = GraphConv(k=self.k, in_dims=self.out_dims[1]*2+3+1, mid_dims = self.out_dims[2], out_dims= self.out_dims[2], layer=2, norm=self.norm, pyramid = self.pyramid[2], dilation = self.dilation, dilation_rate = self.dilation_rate[2]+3,bias = self.bias)
        self.conv3 = nn.Sequential(nn.Conv1d(self.out_dims[2]*2, self.out_dims[2], kernel_size=1, bias=self.bias),
                                  self.bn3,
                                  nn.LeakyReLU(negative_slope=0.2))
        
        self.edge4 = GraphConv(k=self.k, in_dims=self.out_dims[2]*2+3+1, out_dims= self.out_dims[3], layer=1, norm=self.norm, pyramid = self.pyramid[3], dilation = self.dilation, dilation_rate = self.dilation_rate[3],bias = self.bias)
        
        self.linear1 = nn.Linear(self.out_dims[3], 256)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(256, 2)
        
    def forward(self, x):        
        pos = x[:,:3]
        pos_1 = pos
        x1 = self.edge1(x,pos_1)
        #em = x1
        x1_upsample = x1
        x1_cpy = x1
        ######
        bs = torch.arange(x1.size(0)).to(self.device)
        bs = bs.repeat_interleave(x1.size(2))
        pos = pos.permute(2,1,0).contiguous()
        pos = pos.view(pos.size(0)*pos.size(2),-1) #(n_pts*bs),channel
        x1 = x1.permute(2,1,0).contiguous()#n_pts,channel,bs
        x1 = x1.view(x1.size(0)*x1.size(2),-1) #(n_pts*bs),channel

        idx = fps(pos, batch=bs, ratio=0.375) #2048->768
        pos2 = pos[idx]
        pos2 = pos2.view(-1,pos2.size(1),batch_size)
        pos2 = pos2.permute(2,1,0) #batchsize,channel,pts
        pos2_1 = pos2
        x2 = x1[idx] #pts*bs,channel
        x2 = x2.view(-1,x1.size(1),batch_size) #pts,channel,bs
        x2 = x2.permute(2,1,0)#.view(x2.size(2),x2.size(1),x2.size(0))
        #########################
        x2_copy = x2
        x2 = self.edge2(x2,pos2_1)
        
        bs2 = torch.arange(x2.size(0)).to(self.device)
        bs2 = bs2.repeat_interleave(x2.size(2))
        pos2 = pos2.permute(2,1,0).contiguous()
        pos2 = pos2.view(pos2.size(0)*pos2.size(2),-1) #(n_pts*bs),channel
        x2 = x2.permute(2,1,0).contiguous()#view(x1.size(2),x1.size(1),x1.size(0))
        x2 = x2.view(x2.size(0)*x2.size(2),-1) #(n_pts*bs),channel

        idx = fps(pos2, batch=bs2, ratio=1/3) #768 -> 384
        pos3 = pos2[idx]
        pos3 = pos3.view(-1,pos3.size(1),batch_size)
        pos3 = pos3.permute(2,1,0) #batchsize,channel,pts 
        pos3_1 = pos3
        x3 = x2[idx]
        x3 = x3.view(-1,x3.size(1),batch_size) #pts,channel,bs
        x3 = x3.permute(2,1,0)#.view(x3.size(2),x3.size(1),x3.size(0))
        ##########################
        x3_copy = x3
        x3 = self.edge3(x3,pos3_1) #bottleneck
        x3_2 = self.edge3_2(x3_copy, pos3_1)
        x3 = torch.cat((x3,x3_2),1)
        x3 = self.conv3(x3)
        
        bs3 = torch.arange(x3.size(0)).to(self.device)
        bs3 = bs3.repeat_interleave(x3.size(2))
        pos3 = pos3.permute(2,1,0).contiguous()
        pos3 = pos3.view(pos3.size(0)*pos3.size(2),-1) #(n_pts*bs),channel
        x3 = x3.permute(2,1,0).contiguous()#view(x1.size(2),x1.size(1),x1.size(0))
        x3 = x3.view(x3.size(0)*x3.size(2),-1) #(n_pts*bs),channel
        #########################
        idx = fps(pos3, batch=bs3, ratio=1/3) #768 -> 384
        pos4 = pos3[idx]
        pos4 = pos4.view(-1,pos4.size(1),batch_size)
        pos4 = pos4.permute(2,1,0) #batchsize,channel,pts 
        pos4_1 = pos4
        x4 = x3[idx]
        x4 = x4.view(-1,x4.size(1),batch_size) #pts,channel,bs
        x4 = x4.permute(2,1,0)#.view(x3.size(2),x3.size(1),x3.size(0))
        
        x4 = self.edge4(x4, pos4_1)
        em = x4
        
        x3 = x4.max(dim=-1, keepdim=False)[0]
        #em = x3
        
        x = F.leaky_relu(self.bn6(self.linear1(x3)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = self.linear2(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x,em