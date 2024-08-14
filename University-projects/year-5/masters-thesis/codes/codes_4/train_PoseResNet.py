import os
import torch

from dataset.Dataset import StructuredCameraPointCloudDatasetFor2DPoseEstimation
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Trainers.PoseResNetTrainer import Trainer
from Networks.AdaFuse import PoseResNet

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
EXPERIMENT_NAME = 'PoseResNet50'
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-3
LR_MILESTONES = 10, 18, 35
SCHEDULE_RATE = 0.1

if __name__ == '__main__':
    train_dataset = StructuredCameraPointCloudDatasetFor2DPoseEstimation(DATASET_PATH, 'train')
    min_ = train_dataset.min(centered=True)
    max_ = train_dataset.max(centered=True)
    avg = train_dataset.avg()
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

    validation_dataset = StructuredCameraPointCloudDatasetFor2DPoseEstimation(DATASET_PATH, 'val')
    validation_dataloader = SequenceDataLoader(validation_dataset, BATCH_SIZE,
                                               transforms,
                                               shuffle_sequences=False,
                                               shuffle_frames=False,
                                               device=DEVICE)

    network = PoseResNet(n_channels=3, n_joints=22, pretrained=True)
    network = network.to(DEVICE) 
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=LR_MILESTONES, gamma=SCHEDULE_RATE)

    trainer = Trainer(network, train_dataloader,
                      validation_dataloader, optimiser,
                      scheduler, experiment_name=EXPERIMENT_NAME,
                      epoch=14, save_states=True)
    trainer.train(EPOCHS)
