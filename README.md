# GAGCN
GAGCN: Generative Adversarial Graph Convolutional Network for 3D Point Cloud Semantic Segmentation<br />

Key Contributions:<br />
1. Proposed GCN with adversarial learning scheme in 3D point cloud segmentation
2. Utilized Embedding loss for adversarial learning
3. Proposed an effective way for 3D point cloud convolution<br />

# Results<br />
**Instance Average IoU** <br />

PointNet++ | DGCNN | PointCNN | KPConv | Proposed witout Adv | Proposed with Adv
------------ | ------------- | ------------ | ------------- | ------------- | -------------
85.1 | 85.2 | 86.1 | 86.4 | 86.2 | **86.9** 

**Class Average IoU** <br />

PointNet++ | DGCNN | PointCNN | KPConv | Proposed witout Adv | Proposed with Adv
------------ | ------------- | ------------ | ------------- | ------------- | -------------
80.4 | 82.3 | 84.6 | 85.1 | 84.1 | **85.9** 

# Package Installation<br />
This code has been tested on: <br />
ubuntu <br />
torch == 1.7.1 <br />
torch-geometric == 1.7.0 <br />
torch-cluster == 1.5.9 <br />
torch-scatter == 2.0.6 <br />
torch-sparse == 0.6.9 <br />
torch-spline-conv == 1.2.1 <br />
open3d == 0.9.0 <br />
