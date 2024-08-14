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

if __name__ == '__main__':
    """CMU_world = Dataset.StructuredWorldPointCloudDataset(DATASET_PATH, 'all')
    min_ = CMU_world.min(True)
    max_ = CMU_world.max(True)
    avg = CMU_world.avg()
    std = CMU_world.std()
    transforms = Transforms.Compose([Transforms.RemoveHandJoints(),
                                     Transforms.ZeroCenter(avg, transform_skeletons=True),
                                     Transforms.Rescale(min_, max_, -1, 1, transform_skeletons=True),
                                     #Transforms.Standardization(avg, std, transform_skeletons=False),
                                     Transforms.ToTensor(),
                                     Transforms.ToNumpy()])
    
    CMU_world_dataloader = DataLoader.SequenceDataLoader(CMU_world, BATCH, transforms=transforms,
                                              shuffle_sequences=False, shuffle_frames=False, device='cpu')
    #sample = CMU_world[SEQUENCE]
    sample = next(iter(CMU_world_dataloader))
    pt_clouds = sample['point_clouds']
    skeletons = sample['skeletons']
    for skeleton, pt_cloud in zip(skeletons, pt_clouds):
        ax = Visualisation.plot_structured_point_cloud(pt_cloud)
        Visualisation.plot_skeleton(skeleton, ax)
        plt.show()
        break"""

    CMU_camera = Dataset.StructuredCameraPointCloudDataset(DATASET_PATH, 'all')
    min_ = CMU_camera.min(True)
    max_ = CMU_camera.max(True)
    avg = CMU_camera.avg()
    std = CMU_camera.std()
    transforms = Transforms.Compose([Transforms.RemoveHandJoints(),
                                     Transforms.ZeroCenter(avg, transform_skeletons=False),
                                     Transforms.Rescale(min_, max_, -1, 1, transform_skeletons=False),
                                     #Transforms.Standardization(avg, std, transform_skeletons=False),
                                     Transforms.ToTensor(),
                                     Transforms.ToNumpy()])
    
    CMU_camera_dataloader = DataLoader.SequenceDataLoader(CMU_camera, BATCH, transforms=transforms,
                                              shuffle_sequences=False, shuffle_frames=False, device='cpu')
    sample = CMU_camera[SEQUENCE]
    #sample = next(iter(CMU_camera_dataloader))
    pt_clouds = sample['point_clouds']
    skeletons = sample['skeletons']
    offs = sample['2D_skeletons_offsets']
    hms = sample['heat_maps'][:, :, :, :22]
    #mask = hm.eq(1.0)
    deps = sample['2D_skeletons_depths']
    K_inv = sample['intrinsic_matrix_inverted']
    R_inv = sample['rotation_matrix_inverted']
    t_inv = sample['translation_vector_inverted']

    for i in range(len(pt_clouds)):
        hm = hms[i]
        mask = hm == 1.0

        h, w, c = hm.shape
        hm_flat = np.reshape(hm, (w * h, c))
        pos_2d = np.argmax(hm_flat, 0)
        xs = pos_2d % w
        ys = pos_2d // w
        assert np.sum(mask) > 1.0
        off = offs[i]
        x_off = np.repeat(off[:, :, 0, None], 22, -1)[mask]
        y_off = np.repeat(off[:, :, 1, None], 22, -1)[mask]
        dep = deps[i]
        ws = np.repeat(dep[:, :], 22, -1)[mask]
        print(ws.shape)

        pts_2d = np.stack(((xs + x_off) * 4, (ys + y_off) * 4, np.ones(22)), 1) * -ws[:, None]

        pts_3d = (R_inv @ (t_inv[:, None] + (K_inv @ pts_2d.T))).T
        Visualisation.plot_skeleton(pts_3d)
        plt.show()
        break
        
        
    for pt_cloud in pt_clouds:
        ax = Visualisation.plot_structured_point_cloud(pt_cloud)
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

        for i in range(hm.shape[-1]):
            plt.imshow(hm[:, :, i])
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
