import os
import torch

from dataset.Dataset import StructuredCameraPointCloudDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Trainers.CenterNetTrainer import Trainer
from Networks.CenterNet import CenterNet3DPoseEstimation, CenterNet3DPoseEstimationSoftMax, CenterNetDLA#, CenterNetOld

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
EXPERIMENT_NAME = 'CenterNetDLA34PoseEstimationExperimentFix1Schedule(0.5, 5)'
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
SCHEDULE_STEP = 5 #2
SCHEDULE_RATE = 0.5
REDUCTION = 4

if __name__ == '__main__':
    train_dataset = StructuredCameraPointCloudDataset(DATASET_PATH, 'train')
    min_ = train_dataset.minimum_camera_pixel()
    max_ = train_dataset.maximum_camera_pixel()
    avg = train_dataset.average_camera_point()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=False),
                          Rescale(min_, max_, -1, 1, transform_skeletons=False),
                          ToTensor()])
    #train_dataset.transforms = transforms
    train_dataloader = SequenceDataLoader(train_dataset, BATCH_SIZE,
                                          transforms,
                                          shuffle_sequences=True,
                                          shuffle_frames=True,
                                          device=DEVICE)

    validation_dataset = StructuredCameraPointCloudDataset(DATASET_PATH, 'val')
    validation_dataloader = SequenceDataLoader(validation_dataset, BATCH_SIZE,
                                               transforms,
                                               shuffle_sequences=False,
                                               shuffle_frames=False,
                                               device=DEVICE)

    network = CenterNetDLA(n_channels=3, n_joints=22)#CenterNet3DPoseEstimation(n_channels=3, n_joints=22)
    network = network.to(DEVICE) 
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=SCHEDULE_STEP, gamma=SCHEDULE_RATE)

    trainer = Trainer(REDUCTION, network, train_dataloader,
                      validation_dataloader, optimiser,
                      scheduler, experiment_name=EXPERIMENT_NAME,
                      epoch=7, save_states=True)
    trainer.train(EPOCHS)
