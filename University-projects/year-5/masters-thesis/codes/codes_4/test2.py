import numpy as np
import torch
import os
from visualisation import Visualisation
from dataset import Dataset
from dataset import DataLoader
from dataset import Transforms
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join('dataset', 'CMU')
SEQUENCE = 1
BATCH = 32


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

if __name__ == '__main__':
    CMU_camera = Dataset.StructuredCameraPointCloudDataset(DATASET_PATH, 'all')
    min_ = CMU_camera.min(True)
    max_ = CMU_camera.max(True)
    avg = CMU_camera.avg()
    std = CMU_camera.std()
    transforms = Transforms.Compose([Transforms.RemoveHandJoints(),
                                     Transforms.ZeroCenter(avg, transform_skeletons=False),
                                     Transforms.Rescale(min_, max_, -1, 1, transform_skeletons=False),
                                     #Transforms.Standardization(avg, std, transform_skeletons=False),
                                     Transforms.ToTensor()])
    
    CMU_camera_dataloader = DataLoader.SequenceDataLoader(CMU_camera, BATCH, transforms=transforms,
                                              shuffle_sequences=False, shuffle_frames=False, device='cpu')
    sample = next(iter(CMU_camera_dataloader))

    skeleton = sample['skeletons']
    predicted_skeleton = predict_3D_skeleton(sample['heat_maps'], sample['2D_skeletons_offsets'], sample['2D_skeletons_depths'] / 1000,
                        sample['intrinsic_matrix_inverted'],
                        sample['rotation_matrix_inverted'],
                        sample['translation_vector_inverted'],
                        stride=4)
    for gt, pred in zip(skeleton, predicted_skeleton):
        skeleton_numpy = torch.moveaxis(gt, 0, 1).numpy()
        ax = Visualisation.plot_skeleton(skeleton_numpy)
        skeleton_numpy = torch.moveaxis(pred, 0, 1).numpy()
        Visualisation.plot_skeleton(skeleton_numpy, ax)
        plt.show()
    print(MPJPE(skeleton, predicted_skeleton))
    
