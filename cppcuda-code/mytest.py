# import sys

# print(sys.version)
import torch
import torchvision

# print(torch.__version__)  
# print(torch.version.cuda)
# print(torchvision.__version__)
# print(torch.cuda.is_available())
# print(torch.__file__)

feats=torch.rand(4234, 8, 256)
points=torch.rand(4234, 3)*2-1
print("feats: ",feats.shape)
print("points: ",points.shape)
print("points slice: ", points[:, 0:1].shape)
print("points slice: ", points[:, 0].shape)
print("feats slice: ", feats[:, 0].shape)
print("cal: ", (points[:, 0:1]*feats[:, 0]).shape)
print("cal: ", torch.dot(points[:, 0:1], feats[:, 0]).shape)