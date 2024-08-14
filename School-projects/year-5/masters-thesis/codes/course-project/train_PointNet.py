import os
import torch

from dataset.Dataset import PointCloudDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Trainers.PointNetTrainer import Trainer
from Networks.PointNet import PointNetPoseRegression

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
EXPERIMENT_NAME = 'PointNetRegression_Points(1024)_Noise(5)_Rot(0, 0, 0)'
POINTS = 1024
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3

if __name__ == '__main__':
    train_dataset = PointCloudDataset(DATASET_PATH, 'train')
    min_ = train_dataset.minimum_point()
    max_ = train_dataset.maximum_point()
    avg = train_dataset.average_point()
    transforms = Compose([RemoveHandJoints(),
                          RandomSampling(POINTS),
                          AddNormalNoise(5),
                          #RandomRotation(30, 30, 180),
                          ZeroCenter(avg, transform_skeletons=True),
                          Rescale(min_, max_, -1, 1, transform_skeletons=True),
                          ToTensor()])
    train_dataloader = SequenceDataLoader(train_dataset, BATCH_SIZE,
                                          transforms,
                                          shuffle_sequences=True,
                                          shuffle_frames=True,
                                          device=DEVICE)

    validation_dataset = PointCloudDataset(DATASET_PATH, 'val')
    validation_dataloader = SequenceDataLoader(validation_dataset, BATCH_SIZE,
                                               transforms,
                                               shuffle_sequences=False,
                                               shuffle_frames=False,
                                               device=DEVICE)

    network = PointNetPoseRegression(n_channels=3, n_joints=22, device=DEVICE)
    network = network.to(DEVICE) 
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=20, gamma=0.5)

    trainer = Trainer(network, train_dataloader,
                      validation_dataloader, optimiser,
                      scheduler, experiment_name=EXPERIMENT_NAME,
                      epoch=None, save_states=True)
    trainer.train(EPOCHS)
