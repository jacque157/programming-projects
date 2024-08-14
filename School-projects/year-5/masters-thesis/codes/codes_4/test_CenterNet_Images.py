import os
import torch

from dataset.Dataset import ImageDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Networks.CenterNet import CenterNetDLA
from Trainers.CenterNetTrainer import MPJPE, predict_3D_skeleton

from visualisation import Visualisation
import matplotlib.pyplot as plt


DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
EXPERIMENT_NAME = 'CenterNetDLA34PoseEstimationImages'
BATCH_SIZE = 16
REDUCTION = 4
BEST_MODEL = 24 #13


def compute_error(dataloader, network):
    samples = 0
    error_sum = 0
    for i, sample in enumerate(dataloader):
        images = sample['images']
        b, c, h, w = images.shape
        if b == 0:
            continue
        samples += b
        heat_maps, offsets, depths = network(images)
        predicted_3D_skeleton = predict_3D_skeleton(heat_maps, offsets, depths,
                                                    sample['intrinsic_matrix_inverted'],
                                                    sample['rotation_matrix_inverted'],
                                                    sample['translation_vector_inverted'], REDUCTION)
        """for skeleton_predicted, skeleton_target in zip(predicted_3D_skeleton.detach().cpu(), sample['skeletons'].detach().cpu()):
            skeleton_numpy = torch.moveaxis(skeleton_target, 0, 1).numpy()
            ax = Visualisation.plot_skeleton(skeleton_numpy)
            ax.set_title('Ground Truth')
            plt.show()

            skeleton_numpy = torch.moveaxis(skeleton_predicted, 0, 1).numpy()
            ax = Visualisation.plot_skeleton(skeleton_numpy)
            ax.set_title('Prediction')
            plt.show()"""
            
        error = MPJPE(predicted_3D_skeleton, sample['skeletons'])
        print(f'Batch {i}, MPJPE: {error}')
        error_sum += error * b  
    mean_error = error_sum / samples
    return mean_error


if __name__ == '__main__':
    test_dataset = ImageDataset(DATASET_PATH, 'test')
    min_ = test_dataset.min(centered=True)
    max_ = test_dataset.max(centered=True)
    avg = test_dataset.avg()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=False),
                          Rescale(min_, max_, -1, 1, transform_skeletons=False),
                          ToTensor()])
    test_dataloader = SequenceDataLoader(test_dataset, BATCH_SIZE,
                                          transforms,
                                          shuffle_sequences=True,
                                          shuffle_frames=True,
                                          device=DEVICE)
    
    network = CenterNetDLA(n_channels=3, n_joints=22)
    network = network.to(DEVICE)
    path = os.path.join('models', f'{EXPERIMENT_NAME}', f'net_{BEST_MODEL}.pt')
    network.load_state_dict(torch.load(path))
    network.eval()
    with torch.no_grad():
        error = compute_error(test_dataloader, network)
        print(f'MPJPE: {error}')

