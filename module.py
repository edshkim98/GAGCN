import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from attention import *

@torch.jit.script
def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
        
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Mish.html
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return mish(input)
    
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_dims, out_dims, SE=False, norm ='group', kernel_size=1, bias=True):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.kernel_size = kernel_size
        self.bias = bias
        self.norm = norm
        self.se = SE
        
        if self.se:
            self.se_block = SE_Block(self.out_dims)
        self.depthwise = nn.Conv2d(self.in_dims, self.out_dims, kernel_size = self.kernel_size, bias = self.bias, groups = 4)
        #self.pointwise = nn.Conv2d(self.out_dims, self.out_dims, kernel_size=1)
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(self.out_dims)
        else:
            self.bn = nn.GroupNorm(32, self.out_dims)
            
        self.activation = Mish()
        #self.activation2 = Mish()
    def forward(self, x):
        #depthwise
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.activation(out)
        if self.se:
            out = self.se_block(out)
        return out
    
class Conv2d(nn.Module):
    def __init__(self, in_dims, out_dims, SE=False, kernel_size=1, norm = 'group', bias=True):
        super().__init__()
        self.in_dims = in_dims
        self.norm = norm
        self.out_dims = out_dims
        self.kernel_size = kernel_size
        self.bias = bias
        self.se = SE
        
        if self.se:
            self.se_block = SE_Block(self.out_dims)
        self.conv = nn.Conv2d(self.in_dims, self.out_dims, kernel_size = self.kernel_size, bias = self.bias)
        if self.norm == 'batch':
            self.bn = nn.BatchNorm2d(self.out_dims)
        else:
            self.bn = nn.GroupNorm(32, self.out_dims)
        self.activation = Mish()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        if self.se:
            out = self.se_block(out)
        return out
    
class Conv1d(nn.Module):
    def __init__(self, in_dims, out_dims, SE=False, kernel_size=1, norm = 'group', bias=True):
        super().__init__()
        self.in_dims = in_dims
        self.norm = norm
        self.out_dims = out_dims
        self.kernel_size = kernel_size
        self.bias = bias
        self.se = SE
        
        if self.se:
            self.se_block = SE_Block1d(self.out_dims)
        self.conv = nn.Conv1d(self.in_dims, self.out_dims, kernel_size = self.kernel_size, bias = self.bias)
        if self.norm == 'batch':
            self.bn = nn.BatchNorm1d(self.out_dims)
        else:
            self.bn = nn.GroupNorm(32, self.out_dims)
        self.activation = Mish()
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.se:
            out = self.se_block(out)
        return self.activation(out)
    
class GeoEdgeConv(nn.Module):
    def __init__(self,k, in_dims1, in_dims2, out_dims=None, layer=1, norm = 'batch', dilation_rate=1, bias = True, depthwise = True):
        super(GeoEdgeConv, self).__init__()
        self.dilation_rate = dilation_rate
        self.bias = bias
        self.k = k
        self.in_dims1 = in_dims1
        self.in_dims2 = in_dims2
        self.out_dims = out_dims
        self.x = 64
        if self.out_dims == None:
            self.out_dims = self.x
        self.norm = norm
        self.layer = layer
        self.depthwise = depthwise

        if self.depthwise == True:
                
            self.conv1 = DepthwiseConv2d(self.in_dims1, self.out_dims//2, norm=self.norm)
            self.conv2 = DepthwiseConv2d(self.in_dims2, self.out_dims//2, norm=self.norm)
            self.conv3 = Conv2d(self.out_dims//2, self.out_dims//2, SE=True, norm=self.norm)
            self.conv4 = Conv2d(self.out_dims//2, self.out_dims//2, SE=False, norm=self.norm)
            self.conv5 = Conv2d(self.out_dims, self.out_dims, SE=False, norm=self.norm)

        else:
                
            self.conv1 = Conv2d(self.in_dims1, self.out_dims//2, norm=self.norm)
            self.conv2 = Conv2d(self.in_dims2, self.out_dims//2, norm=self.norm)
            self.conv5 = Conv2d(self.out_dims, self.out_dims, SE=True, norm=self.norm)
                            
    def forward(self, x, pos):

        if self.dilation_rate != 1:
            x1_pos, x1, dist_sum = get_graph_feature(x, pos, k=self.k, dilation=True, rate=self.dilation_rate)
        else:
            x1_pos, x1, dist_sum = get_graph_feature(x, pos, k=self.k, dilation=False, rate=1)
           
        if self.depthwise == True:
            x1 = self.conv1(x1)
            x1 = self.conv3(x1)

            x1_pos = self.conv2(x1_pos)
            x1_pos = self.conv4(x1_pos)
            
        else: 
            x1 = self.conv1(x1)
            x1_pos = self.conv2(x1_pos)
        feature = torch.cat((x1,x1_pos),dim=1)
        feature = self.conv5(feature)
        m = nn.AdaptiveMaxPool2d((feature.size(2),1))#.view(batch_size, -1)   
        out = m(feature).squeeze(-1)
                
        return out, dist_sum
    
class EdgeConv(nn.Module):
    def __init__(self,k, in_dims, mid_dims=64, out_dims=None, layer=2, norm = 'batch', dilation_rate=1, bias = True, first =False):
        super(EdgeConv, self).__init__()
        self.dilation_rate = dilation_rate
        self.bias = bias
        self.k = k
        self.first = first
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.x = mid_dims
        
        if self.out_dims == None:
            self.out_dims = self.x
        self.norm = norm
        self.layer = layer
        
        if self.layer == 2:

            self.conv1 = Conv2d(self.in_dims, self.x, norm=self.norm)
            self.conv2 = Conv2d(self.x, self.out_dims, norm=self.norm, SE=True)
                
    def forward(self, x, pos):
        if self.layer == 2:
            if self.dilation_rate != 1:
                x1, dist_sum = get_graph_feature_disc(x, pos, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
            else:
                x1, dist_sum = get_graph_feature_disc(x, pos, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
            x1 = self.conv1(x1)
            x1 = self.conv2(x1)
            x1_max = x1.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
            x1 = x1_max
            
            residual = x1

        return residual, dist_sum