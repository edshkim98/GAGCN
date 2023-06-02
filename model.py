import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from attention import *
from module import *
from graph_utils import *
from utils import *


class AGCN(nn.Module):
    def __init__(self, seg_num_all, k, dilation_rate, norm = 'group', bias = True):
        super(AGCN, self).__init__()
        self.dilation_rate = dilation_rate
        self.bias = bias
        self.norm = norm
        self.seg_num_all = seg_num_all
        self.k = k
        self.device = device
        self.x = 64
        self.edge0 = GeoEdgeConv(k=self.k, in_dims1=6, in_dims2=4, out_dims= 64, layer=1, norm=self.norm, dilation_rate = self.dilation_rate[0],
                                 bias = self.bias, depthwise=False)
        self.edge1 = GeoEdgeConv(k=self.k, in_dims1=self.x*2, in_dims2=4, out_dims= 128, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[1],bias = self.bias, depthwise=False)
        self.edge2 = GeoEdgeConv(k=self.k, in_dims1=self.x*2*2, in_dims2=4, out_dims= 256, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[2],bias = self.bias, depthwise=False)
        self.edge3 = GeoEdgeConv(k=self.k, in_dims1=self.x*2*2*2, in_dims2=4, out_dims= 512, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[3],bias = self.bias, depthwise=False)
        self.edge4 = GeoEdgeConv(k=self.k, in_dims1=2*(256+512), in_dims2=4, out_dims= 256, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[4],bias = self.bias, depthwise=False) #bottleneck
        self.edge5 = GeoEdgeConv(k=self.k, in_dims1=2*(128+256), in_dims2=4,out_dims= 256, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[5],bias = self.bias, depthwise=False)
        self.edge6 = GeoEdgeConv(k=self.k, in_dims1=2*(256), in_dims2=4, out_dims= 256, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[6],bias = self.bias, depthwise=False)              
            
        self.conv4 = Conv1d(256, 128, norm=self.norm, SE=False)
        self.conv5 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=self.bias)

        self.dp = nn.Dropout()        
        
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        pos = x
        pos_1 = pos
        x, dist_sum = self.edge0(x,pos_1)        
        
        pos_1 = pos
        x1, dist_sum = self.edge1(x,pos_1)
        x1_upsample = x1
        x1_cpy = x1
        
        bs = torch.arange(x1.size(0)).to(self.device)
        bs = bs.repeat_interleave(x1.size(2))
        pos_upsample = pos.reshape(3,-1)

        pos2,x2 = IDIS(pos_1,dist_sum, feat=x1, n=512)#pos[idx]

        pos2_1 = pos2

        x2, dist_sum = self.edge2(x2,pos2_1)
        x2_upsample = x2
        
        bs2 = torch.arange(x2.size(0)).to(self.device)
        bs2 = bs2.repeat_interleave(x2.size(2))
        pos2_upsample = pos2.reshape(3,-1)

        pos3,x3 = IDIS(pos2_1, dist_sum, feat=x2, n=256)#pos2[idx]
        pos3_1 = pos3
        pos3_upsample = pos3.reshape(3,-1)

        x3, _ = self.edge3(x3,pos3_1) #bottleneck
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
        x4, _ = self.edge4(x4,pos2_1)
        
        x4_2 = x4.reshape(x4.size(0)*x4.size(2),-1)
        x5 = knn_interpolate(x4_2,pos2.reshape(pos2.size(1),pos2.size(0)),pos.reshape(pos.size(1),pos.size(0)),batch_x = bs2,batch_y = bs)
        x5 = x5.reshape(batch_size,x4_2.size(1),-1)
        x5 = torch.cat((x5,x1_upsample),1)
        x5, _ = self.edge5(x5,pos_1)
    
        x5, _ = self.edge6(x5,pos_1)

        residual = x5
               
        x = self.conv4(x5)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        
        x = self.dp(x)
        x = self.conv5(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x
    
class AGCN_small(nn.Module):
    def __init__(self, seg_num_all, k, dilation_rate, norm = 'group', bias = True):
        super(AGCN_small, self).__init__()
        self.dilation_rate = dilation_rate
        self.bias = bias
        self.norm = norm
        self.seg_num_all = seg_num_all
        self.k = k
        self.device = device
        self.x = 64
        
        self.edge0 = GeoEdgeConv(k=self.k, in_dims1=6, in_dims2=4, out_dims= 64, layer=1, norm=self.norm, dilation_rate = self.dilation_rate[0],
                                 bias = self.bias, depthwise=False)
        self.edge1 = GeoEdgeConv(k=self.k, in_dims1=self.x*2, in_dims2=4, out_dims= 128, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[1],bias = self.bias, depthwise=True)
        self.edge2 = GeoEdgeConv(k=self.k, in_dims1=self.x*2*2, in_dims2=4, out_dims= 256, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[2],bias = self.bias, depthwise=True)
        self.edge3 = GeoEdgeConv(k=self.k, in_dims1=self.x*2*2*2, in_dims2=4, out_dims= 512, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[3],bias = self.bias, depthwise=True)
        self.edge4 = GeoEdgeConv(k=self.k, in_dims1=2*(256+512), in_dims2=4, out_dims= 256, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[4],bias = self.bias, depthwise=True) #bottleneck
        self.edge5 = GeoEdgeConv(k=self.k, in_dims1=2*(128+256), in_dims2=4,out_dims= 256, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[5],bias = self.bias, depthwise=True)
        self.edge6 = GeoEdgeConv(k=self.k, in_dims1=2*(256), in_dims2=4, out_dims= 128, layer=1, norm=self.norm, 
                                 dilation_rate = self.dilation_rate[6],bias = self.bias, depthwise=True)              
            
        self.conv4 = Conv1d(128, 128, norm=self.norm, SE=False)
        self.conv5 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=self.bias)

        self.dp = nn.Dropout()        
        
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        
        pos = x
        pos_1 = pos
        x, dist_sum = self.edge0(x,pos_1)        
        
        pos_1 = pos
        x1, dist_sum = self.edge1(x,pos_1)
        x1_upsample = x1
        x1_cpy = x1
        
        bs = torch.arange(x1.size(0)).to(self.device)
        bs = bs.repeat_interleave(x1.size(2))
        pos_upsample = pos.reshape(3,-1)

        pos2,x2 = IDIS(pos_1,dist_sum, feat=x1, n=512)
        pos2_1 = pos2
        #########################
        x2, dist_sum = self.edge2(x2,pos2_1)
        x2_upsample = x2
        
        bs2 = torch.arange(x2.size(0)).to(self.device)
        bs2 = bs2.repeat_interleave(x2.size(2))
        pos2_upsample = pos2.reshape(3,-1)

        pos3,x3 = IDIS(pos2_1, dist_sum, feat=x2, n=128)
        pos3_1 = pos3
        pos3_upsample = pos3.reshape(3,-1)
        ##########################
        x3, _ = self.edge3(x3,pos3_1) #bottleneck
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
        x4, _ = self.edge4(x4,pos2_1)
        
        x4_2 = x4.reshape(x4.size(0)*x4.size(2),-1)
        x5 = knn_interpolate(x4_2,pos2.reshape(pos2.size(1),pos2.size(0)),pos.reshape(pos.size(1),pos.size(0)),batch_x = bs2,batch_y = bs)
        x5 = x5.reshape(batch_size,x4_2.size(1),-1)
        x5 = torch.cat((x5,x1_upsample),1)
        x5, _ = self.edge5(x5,pos_1)
    
        x5, _ = self.edge6(x5,pos_1)

        residual = x5
               
        x = self.conv4(x5)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        
        x = self.dp(x)
        x = self.conv5(x)                      # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)
        
        return x
    
class Discriminator(nn.Module):
    def __init__(self, k, dilation_rate, n_cls,out_dims,norm='group', bias=True):
        super(Discriminator, self).__init__()
        self.n_cls = n_cls
        self.bias = bias
        self.x = 64
        self.out_dims = out_dims
        self.device = device
        self.k = k
        self.dilation = dilation
        self.dilation_rate = dilation_rate
        self.norm = norm 
        if self.norm == 'batch':
            self.bn2 = nn.BatchNorm1d(self.out_dims[1])
            self.bn3 = nn.BatchNorm1d(self.out_dims[2])
            self.bn6 = nn.BatchNorm1d(256)
        else:
            self.bn2 = nn.GroupNorm(32,self.out_dims[1])
            self.bn3 = nn.GroupNorm(32,self.out_dims[2])
            self.bn6 = nn.GroupNorm(32,256)
            
        self.edge1 = EdgeConv(k=self.k, in_dims=(n_cls+3)*2, mid_dims = self.out_dims[0],out_dims= self.out_dims[0], layer=2, norm=self.norm,
                              dilation_rate = self.dilation_rate[0],bias = self.bias)
        
        self.edge2 = EdgeConv(k=self.k, in_dims=self.out_dims[0]*2, mid_dims = self.out_dims[1],out_dims= self.out_dims[1], layer=2, norm=self.norm,
                              dilation_rate = self.dilation_rate[1],bias = self.bias)
        
        self.edge3 = EdgeConv(k=self.k, in_dims=self.out_dims[1]*2, mid_dims = self.out_dims[2], out_dims= self.out_dims[2], layer=2, norm=self.norm,
                              dilation_rate = self.dilation_rate[2],bias = self.bias)
        
        self.edge4 = EdgeConv(k=self.k, in_dims=self.out_dims[2]*2, out_dims= self.out_dims[3], layer=2, norm=self.norm, 
                              dilation_rate = self.dilation_rate[3],bias = self.bias)
        
        self.linear1 = nn.Linear(self.out_dims[3], 256)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(256, 2)
        
    def forward(self, x):        
        pos = x[:,:3]
        pos_1 = pos
        
        x1, dist_sum = self.edge1(x,pos_1)
        x1_upsample = x1
        x1_cpy = x1
        ######
        bs = torch.arange(x1.size(0)).to(self.device)
        bs = bs.repeat_interleave(x1.size(2))

        pos2,x2 = IDIS(pos_1, dist_sum, feat=x1, n=1024)#pos[idx]#pos2 = pos[idx]
        pos2_1 = pos2
        #########################
        x2_copy = x2
        x2, dist_sum = self.edge2(x2,pos2_1)
        
        bs2 = torch.arange(x2.size(0)).to(self.device)
        bs2 = bs2.repeat_interleave(x2.size(2))

        pos3,x3 = IDIS(pos2_1, dist_sum, feat=x2, n=512)#pos[idx]#pos3 = pos2[idx]
        pos3_1 = pos3
        ##########################
        x3_copy = x3
        x3, dist_sum = self.edge3(x3,pos3_1) #bottleneck
        
        bs3 = torch.arange(x3.size(0)).to(self.device)
        bs3 = bs3.repeat_interleave(x3.size(2))
        #########################
        pos4,x4 = IDIS(pos3_1, dist_sum, feat=x3, n=256)#pos[idx]#pos4 = pos3[idx]
        pos4_1 = pos4
        
        x4, _ = self.edge4(x4, pos4_1)
        em = x4
        
        x3 = x4.max(dim=-1, keepdim=False)[0]
        
        x = Mish()(self.bn6(self.linear1(x3)))#, negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        x = self.linear2(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return x,em