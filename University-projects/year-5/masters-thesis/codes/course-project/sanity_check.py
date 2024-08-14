from dataset.Dataset import *
from dataset.DataLoader import *
from dataset.Transforms import * 
from visualisation.Visualisation import *
from Networks.CenterNet import CenterNetResNet18, CenterNetResNet50, CenterNetDLA, CenterUNet

import os
import cv2 
import torch
from torch import nn

from time import time

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'

LEARNING_RATE = 1e-3
REDUCTION = 4


class TestModel(nn.Module):
    pass

def MPJPE(prediction, ground_truth):
    return torch.mean(torch.sqrt(torch.sum((prediction - ground_truth) ** 2, 1)))

def predict_3D_skeleton(heat_maps, offsets, depths,
                        intrinsic_matrix_inverted,
                        rotation_matrix_inverted,
                        translation_vector_inverted,
                        stride=4):
        b, j, h, w = heat_maps.shape
        heat_maps_flattened = torch.flatten(heat_maps, 2)
        idx = torch.argmax(heat_maps_flattened, 2)[:, :, None]
        xs = idx % w
        ys = idx // w
        positions_2D = torch.stack((xs, ys), 1).squeeze()

        offsets_flattened = torch.flatten(offsets, 2)
        x_offs = offsets_flattened[:, 0, None, :].expand(-1, j, -1)
        y_offs = offsets_flattened[:, 1, None, :].expand(-1, j, -1)
        relevant_x_offs = torch.take_along_dim(x_offs, idx, 2).squeeze()
        relevant_y_offs = torch.take_along_dim(y_offs, idx, 2).squeeze()
        relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 1)

        depths_flattened = -torch.flatten(depths, 2) * 1000
        relevant_depths = torch.take_along_dim(depths_flattened, idx, 2).squeeze()[:, None, :]

        positions_corrected = (positions_2D + relevant_offsets) * stride
        points = torch.concatenate((positions_corrected, torch.ones(b, 1, j, device=positions_corrected.device)), 1)
        points *= relevant_depths
        """points_3D = torch.bmm(rotation_matrix_inverted.expand(b, 3, 3),
                              translation_vector_inverted[None, :, None] + \
                              torch.bmm(intrinsic_matrix_inverted.expand(b, 3, 3), points))"""
        points_flat = torch.reshape(torch.moveaxis(points, 1, 0), (3, b * j))
        points_3D_flat = torch.matmul(rotation_matrix_inverted, torch.matmul(intrinsic_matrix_inverted, points_flat) + translation_vector_inverted[:, None])
        points_3D = torch.moveaxis(torch.reshape(points_3D_flat, (3, b, j)), 0, 1)
        return points_3D


def to_3D(heat_maps, depths,
          intrinsic_matrix_inverted,
          rotation_matrix_inverted,
          translation_vector_inverted,
          stride=4):
    
    b, j, h, w = heat_maps.shape
    heat_maps_flattened = torch.flatten(heat_maps, 2)
    idx = torch.argmax(heat_maps_flattened, 2)[:, :, None]
    xs = idx % w
    ys = idx // w
    
    positions_2D = torch.stack((xs, ys), 1).squeeze() * stride
    points_2D = torch.cat((positions_2D, torch.ones(b, 1, j, device=heat_maps.device)), 1) * -depths * 1000
    points_2D_flat = torch.moveaxis(points_2D, 1, 0).reshape(3, b * j)
    points_3D_flat = torch.matmul(rotation_matrix_inverted, torch.matmul(intrinsic_matrix_inverted, points_2D_flat) + translation_vector_inverted[:, None]) 
    points_3D = torch.moveaxis(torch.reshape(points_3D_flat, (3, b, j)), 0, 1)
    return points_3D

def fun1(heat_map, positions_2d, depths, stride=4):
    b, j, h, w = heat_map.shape

    depth_map = torch.zeros(b, 1, h, w, device=heat_map.device)
    offset_map = torch.zeros(b, 2, h, w, device=heat_map.device)
    for i in range(b):
        for k in range(j):
            c = int(torch.round(positions_2d[i, 0, k] / stride))
            r = int(torch.round(positions_2d[i, 1, k] / stride))
            depth_map[i, 0, r, c] = depths[i, 0, k]
            offset_map[i, 0, r, c] = (positions_2d[i, 0, k] / stride) - c
            offset_map[i, 1, r, c] = (positions_2d[i, 1, k] / stride) - r
    return depth_map, offset_map
    
    

if __name__ == '__main__':
    CMU_camera_pixels = StructuredCameraPointCloudDataset(DATASET_PATH, 'train')
    min_ = CMU_camera_pixels.minimum_camera_pixel()
    max_ = CMU_camera_pixels.maximum_camera_pixel()
    avg = CMU_camera_pixels.average_camera_point()
    transforms = Compose([RemoveHandJoints(), ZeroCenter(avg, transform_skeletons=False), Rescale(min_, max_, transform_skeletons=False), ToTensor()])
    dataloader = SequenceDataLoader(CMU_camera_pixels, 32, transforms, False, False, DEVICE)
    sample = dataloader.__iter__().__next__()


    network = CenterNetDLA(n_channels=3, n_joints=22) #CenterUNet(n_channels=3, n_joints=22, n_layers=50) #CenterNetResNet50(n_channels=3, n_joints=22) #CenterNetDLA(n_channels=3, n_joints=22)#CenterNetResNet18(n_channels=3, n_joints=22)
    network = network.to(DEVICE) 
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.5)

    i = 0
    while True:
        print()
        print(f'Epoch: {i}')
        s = time()
        sample = dataloader.__iter__().__next__()
        pt_cloud = sample['point_clouds']
        #skeleton_2D = sample['skeletons_2D']
        #skeleton_depths = sample['skeletons_depths'] / 1000
        gt_hm = sample['heat_maps']
        gt_dep, gt_off = sample['2D_skeletons_depths'] / 1000, sample['2D_skeletons_offsets'] #fun1(gt_hm, skeleton_2D, skeleton_depths, stride=4)
        print(f'loading time: {time() - s}')

        s = time()
        hm, off, dep = network(pt_cloud)
        print(f'prediction time: {time() - s}')

        s = time()
        l = network.loss(hm, off, dep, gt_hm, gt_off, gt_dep)
        print(f'loss time: {time() - s}')
        print(f'loss: {l.item()}')

        s = time()
        optimiser.zero_grad()
        l.backward()
        optimiser.step()
        scheduler.step()
        print(f'backward time: {time() - s}')

        del l
        del pt_cloud 
        #del skeleton_2D 
        #del skeleton_depths
        torch.cuda.empty_cache()
        
        skeleton_3D = sample['skeletons']
        rot_inv = sample['rotation_matrix_inverted']
        k_inv = sample['intrinsic_matrix_inverted']
        t_inv = sample['translation_vector_inverted']
        predicted_skeletons = predict_3D_skeleton(gt_hm, gt_off, gt_dep,
                                                  k_inv, rot_inv, t_inv,
                                                  stride=4)
        acc = MPJPE(predicted_skeletons, skeleton_3D)
        print(f'1. best possible acc: {acc.item()}')

        
        """predicted_skeletons = to_3D(gt_hm, skeleton_depths,
                                    k_inv, rot_inv, t_inv, 4)"""
        
        s = time()
        predicted_skeletons = predict_3D_skeleton(hm, off, dep,
                                                  k_inv, rot_inv, t_inv,
                                                  stride=4)
        print(f'reprojection time: {time() - s}')
        
        acc = MPJPE(predicted_skeletons, skeleton_3D)
        print(f'2. predicted acc: {acc.item()}')
        
        """predicted_skeletons = to_3D(gt_hm, skeleton_depths,
                                    k_inv, rot_inv, t_inv, 4)   
        acc = MPJPE(predicted_skeletons, skeleton_3D)
        print(f'2. acc: {acc.item()}')"""
        i += 1

        del acc       
        del hm
        del off
        del dep
        del gt_hm
        del gt_off
        del gt_dep
        del predicted_skeletons
        del skeleton_3D
        del rot_inv
        del k_inv
        del t_inv   
        del sample
        torch.cuda.empty_cache()
        
