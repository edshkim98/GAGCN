import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def knn(x, k):

    x1 = x.permute(0,2,1)
    dist, idx = torch.cdist(x1,x1,p=2).topk(k=k,dim=-1,largest=False)
    dist_sum = dist#.sum(axis=2)
    return idx, dist_sum

def knn_dilate(x, k, rate, dilation_type):
    batch_size = x.size(0)
    k2 = rate*k

    x1 = x.permute(0,2,1)
    dist, idx = torch.cdist(x1,x1,p=2).topk(k=k2,dim=-1,largest=False)
    dist_sum = dist#.sum(axis=2)
    idx = idx.cpu().detach().numpy()
        
    if dilation_type == 'order':
    #Ordered sampling
        idx = idx[:, :, ::rate]
    else:
    #Random sampling
        idx_i = np.random.choice(idx.shape[2],k,replace=False)
        idx = idx[:, :, idx_i]
    idx = torch.tensor(idx).to(device)
    return idx, dist_sum

def get_graph_feature(x, pos, k, dilation=True, rate = 2,dilation_type='order',first=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    x_pos = pos.view(batch_size, -1, num_points)

    if dilation == True:
        idx, dist_sum = knn_dilate(pos, k=k, rate= rate, dilation_type = dilation_type)
    else:
        idx, dist_sum = knn(pos, k=k)   # (batch_size, num_points, k)

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
    
    if first:
        feature = torch.cat((feature_pos-x_pos, feature_add, feature-x, x), dim=3).permute(0,3,1,2).contiguous()
        return feature
    
    feature_pos = torch.cat((feature_pos-x_pos, feature_add), dim=3).permute(0,3,1,2).contiguous()
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature_pos, feature, dist_sum      # (batch_size, 3*num_dims+1, num_points, k)

def get_graph_feature_disc(x, pos, k, dilation=True, rate = 2,dilation_type='order'):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    x_pos = pos.view(batch_size, -1, num_points)

    idx, dist_sum = knn(x, k=k)   # (batch_size, num_points, k)

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
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature, dist_sum      # (batch_size, 3*num_dims+1, num_points, k)