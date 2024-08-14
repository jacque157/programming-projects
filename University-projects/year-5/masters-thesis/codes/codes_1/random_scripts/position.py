"""import numpy as np


def get_joints_position(heatmap):
    b, j, d, h, w = heatmap.shape  #z -> h, d ->y, w -> x

    heatmap_exp = np.exp(heatmap)
    sum_ = np.sum(heatmap_exp, (2, 3, 4))[:, :, np.newaxis, np.newaxis, np.newaxis]
    softmax = heatmap_exp / sum_

    idx = np.indices((h, w, d))
    idx = np.moveaxis(idx, 0, -1)

    coords = idx #/ [h - 1, w - 1, d - 1]
    #coords *= 100
    #coords -= 200

    coords = coords[np.newaxis, np.newaxis] * softmax[:, :, :, :, :, np.newaxis]
    coords = np.sum(coords, (2, 3, 4))

    return coords
    
    heatmap_vector = np.reshape(heatmap, (b, j, d * h * w))

    heatmap_exp = np.exp(heatmap_vector)
    sum_ = np.sum(heatmap_exp, 2)[:, :, np.newaxis]
    softmax = heatmap_exp / sum_

    indices = np.arange(h * w * d)[np.newaxis, np.newaxis]
    indices = np.sum(indices * softmax, 2)
    indices = np.uint8(np.floor(indices))

    indices_3d = np.unravel_index(indices, (d, h, w))
    #coords = 
    
    return indices

h = np.zeros((4, 12, 50, 50, 50))
h[:,0, 0, 0, 49] = 10
p =get_joints_position(h)
print(p[:, 0])
"""

import torch
import torch.nn as nn
import time 

def soft_argmax(voxels):
    # https://github.com/Fdevmsy/PyTorch-Soft-Argmax/blob/master/soft-argmax.py
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    assert voxels.dim()==5
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1#1000.0 
    N,C,H,W,D = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N,C,-1)*alpha,dim=2)
    soft_max = soft_max.view(voxels.shape)
    indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0).to('cuda')
    indices_kernel = indices_kernel.view((H,W,D))
    conv = soft_max*indices_kernel
    indices = conv.sum(2).sum(2).sum(2)
    z = indices%D
    y = (indices/D).floor()%W
    x = (((indices/D).floor())/W).floor()%H
    coords = torch.stack([x,y,z],dim=2)
    return coords


H, W, D = 90, 90, 90
numbers = torch.arange(H * W * D).reshape(H, W, D).to('cuda')
indices = torch.zeros(3, H, W, D).to('cuda')
indices[2, :, :, :] = numbers % D
indices[1, :, :, :] = (numbers // D) % W
indices[0, :, :, :] = ((numbers // D) // W) % H

def my_argmax(voxels):
    N,C,H,W,D = voxels.shape
    soft_max = nn.functional.softmax(voxels.view(N,C,-1),dim=2)
    
    

    mult = indices.view(1, 1, 3, -1) * soft_max.view(N, C, 1, -1)
    points = torch.sum(mult, dim=3)
    return points

if __name__ == "__main__":
    #voxel = torch.randn(1,2,2,3,3) # (batch_size, channel, H, W, depth)
    voxel = torch.zeros(16,24,90,90,90).to('cuda')
    voxel[:, 0, 0, 0, 49] = 1000
    voxel[:, 0, 0, 0, 48] = 900
    voxel[:, 0, 0, 0, 47] = 900
    voxel[:, 0, 0, 0, 46] = 900
    s = time.time()
    coords = soft_argmax(voxel)
    print(f'Time taken: {time.time() - s}')
    print(coords[:, 0])
    s = time.time()
    a = my_argmax(voxel)
    print(f'Time taken: {time.time() - s}')
    print(a[:, 0])
