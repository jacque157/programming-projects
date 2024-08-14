import os
import torch
from matplotlib import pyplot as plt

from dataset.Dataset import ImageDataset, StructuredCameraPointCloudDataset
from dataset.DataLoader import SequenceDataLoader, SingleBatchSequenceDataLoader, PartitionSequenceDataLoader
from dataset.Transforms import *
from Networks.GoogleNet import GoogleNet
from visualisation.Visualisation import plot_skeleton, plot_structured_point_cloud
from Trainers.CenterNetTrainer import predict_3D_skeleton, MPJPE


DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
SCHEDULE_STEP = 15
SCHEDULE_RATE = 0.5

"""
LR Scheduler ADAM
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
SCHEDULE_STEP = 15
SCHEDULE_RATE = 0.5
"""
# epoch: 99, loss: 0.000, absolute MPJPE: 20.134, 

"""
Pretrained
LR Scheduler ADAM
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
SCHEDULE_STEP = 15
SCHEDULE_RATE = 0.5
"""
# epoch: 99, loss: 0.000, absolute MPJPE: 9.638, 

"""
LR Cosine ADAM
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
"""
# epoch: 99, loss: 0.000, absolute MPJPE: 11.844, 

"""
LR Cosine ADAM
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
"""
# epoch: 99, loss: 0.000, absolute MPJPE: 1.189, 

"""
Pretrained
LR Cosine ADAM
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
"""
# epoch: 99, loss: 0.000, absolute MPJPE: 1.266, 

"""
Pretrained
LR Cosine AMSGrad
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
"""
# epoch: 99, loss: 0.000, absolute MPJPE: 3.075, 

"""
Pretrained
LR Cosine AMSGrad
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 5e-4
"""
# epoch: 99, loss: 0.000, absolute MPJPE: 1.543, 

"""
Modded GoogleNet
Pretrained
LR Cosine 
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 5e-4
"""
# epoch: 99, loss: 0.000, absolute MPJPE: 5.100, 

"""
Modded GoogleNet
Pretrained
LR Cosine 
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
"""
# epoch: 99, loss: 0.000, absolute MPJPE: 10.612, 

def rescale(skeletons, mean, min_, max_, a=-1, b=1):
    skeletons = (skeletons - a) * (max_[None, :, None] - min_[None, :, None]) / (b - a)
    skeletons += min_[None, :, None]
    skeletons += mean[None, :, None]
    return skeletons

def project_camera_skeletons(skeletons, rotation_matrix_inverted, translation_vector_inverted):
    #b, c, j = skeletons.shape
    points_3D = torch.matmul(rotation_matrix_inverted, skeletons + translation_vector_inverted[None, :, None])
    return points_3D
    """points_flat = torch.reshape(torch.moveaxis(points, 1, 0), (3, b * j))
    points_3D_flat = torch.matmul(rotation_matrix_inverted, torch.matmul(intrinsic_matrix_inverted, points_flat) + translation_vector_inverted[:, None])
    points_3D = torch.moveaxis(torch.reshape(points_3D_flat, (3, b, j)), 0, 1)"""

def MPJPE(prediction, target):
    return torch.mean(torch.sqrt(torch.sum((prediction - target) ** 2, 1))) 
    
if __name__ == '__main__':
    dataset = StructuredCameraPointCloudDataset(DATASET_PATH, 'train', camera_space_skeletons=True)
    min_ = dataset.min(centered=True)
    max_ = dataset.max(centered=True)
    avg = dataset.avg()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=True),
                          Rescale(min_, max_, -1, 1, transform_skeletons=True),
                          ToTensor()])

    dataloader = SingleBatchSequenceDataLoader(dataset, BATCH_SIZE,
                                               index=0,
                                               transforms=transforms,
                                               shuffle_frames=False,
                                               device=DEVICE)
    
    network = GoogleNet(n_channels=3, n_joints=22, pretrained=True)
    network = network.to(DEVICE) 
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    #optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, amsgrad=True)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=SCHEDULE_STEP, gamma=SCHEDULE_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, EPOCHS)
    min_ = torch.from_numpy(dataset.min(centered=True)).to(DEVICE) 
    max_ = torch.from_numpy(dataset.max(centered=True)).to(DEVICE) 
    avg = torch.from_numpy(dataset.avg()).to(DEVICE) 
    
    for e in range(EPOCHS):
        for sample in dataloader:
            sequences = sample['point_clouds'] #sample['images']
            skeletons_targets = sample['skeletons']
            predicted_skeleton = network(sequences)
            loss = network.loss(predicted_skeleton, skeletons_targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            predicted_skeleton = rescale(predicted_skeleton, avg, min_, max_, a=-1, b=1)
            predicted_skeleton_3D = project_camera_skeletons(predicted_skeleton,  sample['rotation_matrix_inverted'].double(),  sample['translation_vector_inverted'].double())

            skeletons_targets = rescale(skeletons_targets, avg, min_, max_, a=-1, b=1)
            skeletons_targets_3D = project_camera_skeletons(skeletons_targets,  sample['rotation_matrix_inverted'].double(),  sample['translation_vector_inverted'].double())

            acc = MPJPE(predicted_skeleton_3D, skeletons_targets_3D)
            print(f'epoch: {e}, loss: {loss:.03f}, absolute MPJPE: {acc:.03f}, ')

        scheduler.step()
        
