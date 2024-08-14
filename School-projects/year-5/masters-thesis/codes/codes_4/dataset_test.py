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

_PARENTS = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4, 5, 6,  7,  8,  9,   9,  9, 12, 13, 14, 16, 17, 18, 19])
                    
def relative_pose(absolute_skeletons, device):
    parents = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4, 5, 6,  7,  8,  9,   9,  9, 12, 13, 14, 16, 17, 18, 19], device=absolute_skeletons.device)
    absolute_skeletons[:, :, 1:] -= absolute_skeletons[:, :, parents[1:]]
    return absolute_skeletons

def absolute_pose(relative_skeletons, device):
    relative_skeletons[:, :, [1, 2, 3]] += relative_skeletons[:, :, 0, None]
    relative_skeletons[:, :, [4, 5, 6]] += relative_skeletons[:, :, [1, 2, 3]]
    relative_skeletons[:, :, [7, 8, 9]] += relative_skeletons[:, :, [4, 5, 6]]
    relative_skeletons[:, :, [10, 11, 12, 13, 14]] += relative_skeletons[:, :, [7, 8, 9, 9, 9]]
    relative_skeletons[:, :, [15, 16, 17]] += relative_skeletons[:, :, [12, 13, 14]]
    relative_skeletons[:, :, [18, 19]] += relative_skeletons[:, :, [16, 17]]
    relative_skeletons[:, :, [20, 21]] += relative_skeletons[:, :, [18, 19]]
    return relative_skeletons

if __name__ == '__main__':
    CMU_world = Dataset.StructuredWorldPointCloudDataset(DATASET_PATH, 'all')
    min_ = CMU_world.min(True)
    max_ = CMU_world.max(True)
    avg = CMU_world.avg()
    std = CMU_world.std()
    transforms = Transforms.Compose([Transforms.RemoveHandJoints(),
                                     Transforms.ZeroCenter(avg, transform_skeletons=True),
                                     Transforms.Rescale(min_, max_, -1, 1, transform_skeletons=True),
                                     #Transforms.Standardization(avg, std, transform_skeletons=False),
                                     Transforms.ToTensor(),
                                     ])
    numpy = Transforms.ToNumpy()
    CMU_world_dataloader = DataLoader.SequenceDataLoader(CMU_world, BATCH, transforms=transforms,
                                              shuffle_sequences=False, shuffle_frames=False, device='cpu')
    #sample = CMU_world[SEQUENCE]
    sample = next(iter(CMU_world_dataloader))
    pt_clouds = sample['point_clouds']
    skeletons = sample['skeletons']
    relative_skeletons = relative_pose(skeletons, device='cpu')
    skeletons = absolute_pose(relative_skeletons, device='cpu')
    skeletons = torch.movedim(skeletons, 1, 2).numpy()
    pt_clouds = torch.movedim(pt_clouds, 1, -1).numpy()
    for skeleton, pt_cloud in zip(skeletons, pt_clouds):
        ax = Visualisation.plot_structured_point_cloud(pt_cloud)
        Visualisation.plot_skeleton(skeleton, ax)
        plt.show()
        break

    CMU_camera = Dataset.StructuredCameraPointCloudDataset(DATASET_PATH, 'all', camera_space_skeletons=True)
    min_ = CMU_camera.min(True)
    max_ = CMU_camera.max(True)
    avg = CMU_camera.avg()
    std = CMU_camera.std()
    transforms = Transforms.Compose([Transforms.RemoveHandJoints(),
                                     Transforms.ZeroCenter(avg, transform_skeletons=True),
                                     Transforms.Rescale(min_, max_, -1, 1, transform_skeletons=True),
                                     #Transforms.Standardization(avg, std, transform_skeletons=False),
                                     Transforms.ToTensor(),
                                     Transforms.ToNumpy()])
    
    CMU_camera_dataloader = DataLoader.SequenceDataLoader(CMU_camera, BATCH, transforms=transforms,
                                              shuffle_sequences=False, shuffle_frames=False, device='cpu')
    #sample = CMU_world[SEQUENCE]
    sample = next(iter(CMU_camera_dataloader))
    pt_clouds = sample['point_clouds']
    skeletons = sample['skeletons']
    for skeleton, pt_cloud in zip(skeletons, pt_clouds):
        ax = Visualisation.plot_structured_point_cloud(pt_cloud)
        Visualisation.plot_skeleton(skeleton, ax)
        plt.show()
        break

    CMU_imgs = Dataset.ImageDataset(DATASET_PATH, 'all')
    min_ = CMU_imgs.min(True)
    max_ = CMU_imgs.max(True)
    avg = CMU_imgs.avg()
    std = CMU_imgs.std()
    transforms = Transforms.Compose([Transforms.RemoveHandJoints(),
                                     Transforms.ZeroCenter(avg, transform_skeletons=False),
                                     Transforms.Rescale(min_, max_, -1, 1, transform_skeletons=False),
                                     #Transforms.Standardization(avg, std, transform_skeletons=False),
                                     Transforms.ToTensor(),
                                     Transforms.ToNumpy()])
    
    CMU_imgs_dataloader = DataLoader.SequenceDataLoader(CMU_imgs, BATCH, transforms=transforms,
                                              shuffle_sequences=False, shuffle_frames=False, device='cpu')
    #sample = CMU_world[SEQUENCE]
    sample = next(iter(CMU_imgs_dataloader))
    imgs = sample['images']
    heat_maps = sample['heat_maps']
    offset_maps = sample['2D_skeletons_offsets']
    depths = sample['2D_skeletons_depths']
    for img, hm, off, dep in zip(imgs, heat_maps, offset_maps, depths):
        plt.imshow(img)
        plt.show()

        plt.imshow(np.sum(hm, -1))
        plt.show()

        plt.imshow(off[:, :, 0])
        plt.show()

        plt.imshow(off[:, :, 1])
        plt.show()

        plt.imshow(dep)
        plt.show()
    
        break

    images = CMU_imgs[0]['images']
    Visualisation.animate_images(images)
    plt.show()
