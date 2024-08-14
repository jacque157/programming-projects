import os
import torch

from dataset.Dataset import StructuredCameraPointCloudDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Trainers.RegressionTrainer import Trainer
from Networks.ResNet import build_ResNet

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:1'
LAYERS = 34
EXPERIMENT_NAME = f'ResNet{LAYERS}Regression'
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 5e-4
SCHEDULE_STEP = 3
SCHEDULE_RATE = 0.5

if __name__ == '__main__':
    train_dataset = StructuredCameraPointCloudDataset(DATASET_PATH, 'train')
    min_ = train_dataset.min(centered=True)
    max_ = train_dataset.max(centered=True)
    avg = train_dataset.avg()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=True),
                          Rescale(min_, max_, -1, 1, transform_skeletons=True),
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

    network = build_ResNet(layers=LAYERS, in_channels=3, n_joints=22, pretrained=True)
    network = network.to(DEVICE) 
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=SCHEDULE_STEP, gamma=SCHEDULE_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, EPOCHS)

    trainer = Trainer(network, train_dataloader,
                      validation_dataloader, optimiser,
                      scheduler, experiment_name=EXPERIMENT_NAME,
                      epoch=12, save_states=True)
    trainer.train(EPOCHS)
