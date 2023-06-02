import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

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

def read_pts(file):
    verts = np.genfromtxt(file)
    #return utils.cent_norm(verts)
    return verts

def read_seg(file):
    verts = np.genfromtxt(file, dtype= (int))
    return verts

def sample_3000(pts, pts_cat):    
    res1 = np.concatenate((pts, pts_cat.reshape(pts_cat.shape[0],1)), axis= 1) #pts*4
    idx = np.random.choice(len(res1),2048,replace=True)
    res = res1[idx,:]
    #res = np.asarray(random.choices(res1, weights=None, cum_weights=None, k=2048))
    images = res[:, 0:3]
    categories = res[:, 3]
    categories-=np.ones(categories.shape) #making the label 1 or 0
    return images, categories

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0)
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud
    
def sampling(pts, pts_cat):    
    res1 = np.concatenate((pts,np.reshape(pts_cat, (pts_cat.shape[0], 1))), axis= 1)
    res = np.asarray(random.choices(res1, weights=None, cum_weights=None, k=2048))
    images = res[:, 0:3]
    categories = res[:, 3]
    categories-=np.ones(categories.shape)
    return images, categories

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
    
def knn_return_dist(x,k):
    x1 = x.permute(0,2,1)
    dist, idx = torch.cdist(x1,x1,p=2).topk(k=k,dim=-1,largest=False)
    dist = torch.sum(dist,axis=2)
    return dist

def IDIS(pts, dist_sum, feat=None, n=2048, k=16):
    #bs*c*n_pts
    dist = dist_sum[:,:,:k].sum(axis=2)#knn_return_dist(pts, k)
    importance = dist**2
    
    lst = []
    feat_lst = []
    for i in range(importance.shape[0]):
        idx = np.random.choice(pts.size()[-1],n, replace=True)
        lst.append(pts[i][:,idx])
        if feat is not None:
            feat_lst.append(feat[i][:,idx])
    return torch.stack(lst).to(device),torch.stack(feat_lst).to(device)

# def knn_return_dist2(x1,k):
#     x1 = torch.tensor(np.expand_dims(x1,axis=0))
#     dist, idx = torch.cdist(x1,x1,p=2).topk(k=k,dim=-1,largest=False)
#     dist = torch.sum(dist,axis=2)
#     return np.array(dist)
# class Idis(object):
#     def __call__(self, pts, label, n=2048):
#         dist = knn_return_dist2(pts,k=16)
#         dist = dist.squeeze(0)
#         importance = dist**2
#         importance = np.divide(importance,importance.sum())  # normalize
        
#         idx = np.random.choice(pts.shape[0],n,p=importance, replace=True)
#         return pts[idx], label[idx]