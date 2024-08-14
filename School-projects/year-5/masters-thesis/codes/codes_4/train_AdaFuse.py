import os
import torch

from dataset.Dataset import MultiviewStructuredCameraPointCloudDataset as MultiviewDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Trainers.AdaFuseTrainer import Trainer
from Networks.AdaFuse import PoseResNet, AdaFuse


DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
EXPERIMENT_NAME = 'AdaFuse'
POSE_NET_PATH = os.path.join('models', 'PoseResNet50', 'net_20.pt')
BATCH_SIZE = 16 // 16
EPOCHS = 10
LEARNING_RATE = 1e-4
LR_MILESTONES = 18, 35
SCHEDULE_RATE = 0.1
REDUCTION = 4
HEAT_MAPS_WIDTH = 512
HEAT_MAPS_HEIGHT = 512

if __name__ == '__main__':
    train_dataset = MultiviewDataset(DATASET_PATH, 'train')
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

    validation_dataset = MultiviewDataset(DATASET_PATH, 'val')
    validation_dataloader = SequenceDataLoader(validation_dataset, BATCH_SIZE,
                                               transforms,
                                               shuffle_sequences=False,
                                               shuffle_frames=False,
                                               device=DEVICE)

    pose_network = PoseResNet(n_channels=3, n_joints=22, pretrained=False)
    pose_network.load_state_dict(torch.load(POSE_NET_PATH))
    network = AdaFuse(n_channels=3, n_joints=22, PoseResNet=pose_network,
                      heatmap_dimensions=(HEAT_MAPS_HEIGHT // REDUCTION, HEAT_MAPS_WIDTH // REDUCTION))
    network = network.to(DEVICE)
    
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimiser, milestones=LR_MILESTONES, gamma=SCHEDULE_RATE)

    trainer = Trainer(network, train_dataloader,
                      validation_dataloader, optimiser,
                      scheduler, experiment_name=EXPERIMENT_NAME,
                      epoch=None, save_states=True)
    trainer.train(EPOCHS)
