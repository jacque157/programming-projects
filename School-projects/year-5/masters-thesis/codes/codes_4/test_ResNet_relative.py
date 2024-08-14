import os
import torch

from dataset.Dataset import StructuredCameraPointCloudDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Trainers.RegressionTrainer import rescale, MPJPE, project_camera_skeletons
from Networks.ResNet import build_ResNet

from visualisation import Visualisation
import matplotlib.pyplot as plt


DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
LAYERS = 34
EXPERIMENT_NAME = f'ResNet{LAYERS}Regression'
BATCH_SIZE = 16
BEST_MODEL = 5


def compute_error(dataloader, network):
    samples = 0
    error_sum = 0
    for i, sample in enumerate(dataloader):
        point_clouds = sample['point_clouds']
        b, c, h, w = point_clouds.shape
        if b == 0:
            continue
        samples += b
        predicted_skeleton = network(point_clouds)
        predicted_skeleton = rescale(predicted_skeleton, sample['center'],
                                     sample['min_'], sample['max_'],
                                     a=sample['a'], b=sample['b'])
        predicted_skeleton_3D = project_camera_skeletons(predicted_skeleton,
                                                         sample['rotation_matrix_inverted'].double(),
                                                         sample['translation_vector_inverted'].double())

        skeletons_targets = rescale(sample['skeletons'], sample['center'] ,
                                    sample['min_'], sample['max_'],
                                    a=sample['a'], b=sample['b'])
        skeletons_targets_3D = project_camera_skeletons(skeletons_targets,
                                                        sample['rotation_matrix_inverted'].double(),
                                                        sample['translation_vector_inverted'].double())
        """for skeleton_predicted, skeleton_target in zip(predicted_skeleton_3D.detach().cpu(), skeletons_targets.detach().cpu()):
            skeleton_numpy = torch.moveaxis(skeleton_target, 0, 1).numpy()
            ax = Visualisation.plot_skeleton(skeleton_numpy)
            ax.set_title('Ground Truth')
            plt.show()

            skeleton_numpy = torch.moveaxis(skeleton_predicted, 0, 1).numpy()
            ax = Visualisation.plot_skeleton(skeleton_numpy)
            ax.set_title('Prediction')
            plt.show()"""

            
        error = MPJPE(predicted_skeleton_3D, skeletons_targets_3D)
        if i % 20 == 0:
            print(f'Batch {i}, MPJPE: {error}')
        error_sum += error * b  
    mean_error = error_sum / samples
    return mean_error


if __name__ == '__main__':
    test_dataset = StructuredCameraPointCloudDataset(DATASET_PATH, 'test')
    min_ = test_dataset.min(centered=True)
    max_ = test_dataset.max(centered=True)
    avg = test_dataset.avg()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=True),
                          Rescale(min_, max_, -1, 1, transform_skeletons=True),
                          ToTensor()])
    test_dataloader = SequenceDataLoader(test_dataset, BATCH_SIZE,
                                          transforms,
                                          shuffle_sequences=False,
                                          shuffle_frames=False,
                                          device=DEVICE)

    network = build_ResNet(layers=LAYERS, in_channels=3, n_joints=22, pretrained=True)
    network = network.to(DEVICE) 
    path = os.path.join('models', f'{EXPERIMENT_NAME}', f'net_{BEST_MODEL}.pt')
    network.load_state_dict(torch.load(path, map_location='cuda:0'))
    network.eval()
    with torch.no_grad():
        error = compute_error(test_dataloader, network)
        print(f'MPJPE: {error}')
