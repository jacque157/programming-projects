import os
import torch
from matplotlib import pyplot as plt

from dataset.Dataset import ImageDataset
from dataset.DataLoader import SequenceDataLoader, SingleBatchSequenceDataLoader, PartitionSequenceDataLoader
from dataset.Transforms import *
from visualisation.Visualisation import plot_skeleton

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
BATCH_SIZE = 16

if __name__ == '__main__':
    dataset = ImageDataset(DATASET_PATH, 'train')
    min_ = dataset.min(centered=True)
    max_ = dataset.max(centered=True)
    avg = dataset.avg()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=False),
                          Rescale(min_, max_, -1, 1, transform_skeletons=False),
                          ToTensor(),
                          ToNumpy()])
    dataloader = SequenceDataLoader(dataset, BATCH_SIZE,
                                    transforms,
                                    shuffle_sequences=False,
                                    shuffle_frames=False,
                                    device=DEVICE)
    """dataloader = SingleBatchSequenceDataLoader(dataset, BATCH_SIZE,
                                               index=0,
                                               transforms=transforms,
                                               shuffle_frames=False,
                                               device=DEVICE)
    for sample in dataloader:
        skeletons = sample['skeletons']
        print('single')
        print(len(skeletons))
        for skeleton in skeletons:
            ax = plot_skeleton(skeleton)
            plt.show()"""
        
    dataloader = PartitionSequenceDataLoader(dataset, BATCH_SIZE,
                                             maximum=10,
                                             transforms=transforms,
                                             shuffle_sequences=False,
                                             shuffle_frames=False,
                                             device=DEVICE)
    for sample in dataloader:
        skeletons = sample['skeletons']
        print('partition')
        print(len(skeletons))
        """for skeleton in skeletons:
            ax = plot_skeleton(skeleton)
            plt.show()"""
        
