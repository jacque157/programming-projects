import os
import torch
from matplotlib import pyplot as plt

from dataset.Dataset import ImageDataset
from dataset.DataLoader import SequenceDataLoader, SingleBatchSequenceDataLoader, PartitionSequenceDataLoader
from dataset.Transforms import *
from Networks.CenterNet import CenterNetDLA
from visualisation.Visualisation import plot_skeleton
from Trainers.CenterNetTrainer import predict_3D_skeleton, MPJPE


DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
SCHEDULE_STEP = 15
SCHEDULE_RATE = 0.5
REDUCTION = 4
#4, 2 last epoch loss: 0.195, absolute MPJPE: 66.043,
#2, 4 (last) epoch: 99, loss: 2.205, absolute MPJPE: 342.413,
#1, 1 (last) epoch: 99, loss: 3.542, absolute MPJPE: 537.049, 
if __name__ == '__main__':
    dataset = ImageDataset(DATASET_PATH, 'train')
    min_ = dataset.min(centered=True)
    max_ = dataset.max(centered=True)
    avg = dataset.avg()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=False),
                          Rescale(min_, max_, -1, 1, transform_skeletons=False),
                          ToTensor()])

    dataloader = SingleBatchSequenceDataLoader(dataset, BATCH_SIZE,
                                               index=0,
                                               transforms=transforms,
                                               shuffle_frames=False,
                                               device=DEVICE)
    
    network = CenterNetDLA(n_channels=3, n_joints=22)
    network = network.to(DEVICE) 
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=SCHEDULE_STEP, gamma=SCHEDULE_RATE)

    for e in range(EPOCHS):
        for sample in dataloader:
            sequences = sample['images']
            heat_maps_targets = sample['heat_maps']
            target_offsets = sample['2D_skeletons_offsets']
            target_depths = sample['2D_skeletons_depths'] / 1000
            skeletons_targets = sample['skeletons']  
            heat_maps, offsets, depths = network(sequences)
            loss = network.loss(heat_maps,
                                offsets,
                                depths,
                                heat_maps_targets,
                                target_offsets,
                                target_depths)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            predicted_3D_skeleton = predict_3D_skeleton(heat_maps, offsets, depths,
                                                                sample['intrinsic_matrix_inverted'],
                                                                sample['rotation_matrix_inverted'],
                                                                sample['translation_vector_inverted'], 4)
            acc = MPJPE(predicted_3D_skeleton, skeletons_targets)
            print(f'epoch: {e}, loss: {loss:.03f}, absolute MPJPE: {acc:.03f}, ')
        scheduler.step()
        
