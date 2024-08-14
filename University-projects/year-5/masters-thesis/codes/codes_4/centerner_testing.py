import os
import torch
from matplotlib import pyplot as plt

from dataset.Dataset import StructuredCameraPointCloudDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Trainers.CenterNetTrainer import predict_3D_skeleton, MPJPE
from Networks.CenterNet import CenterNetDLA
from visualisation.Visualisation import plot_skeleton

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
BATCH_SIZE = 16
MODEL_PATH = os.path.join('models', 'CenterNetDLA34PoseEstimation', 'net_13.pt')

if __name__ == '__main__':
    dataset = StructuredCameraPointCloudDataset(DATASET_PATH, 'test')
    min_ = dataset.min(centered=True)
    max_ = dataset.max(centered=True)
    avg = dataset.avg()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=False),
                          Rescale(min_, max_, -1, 1, transform_skeletons=False),
                          ToTensor()])
    #train_dataset.transforms = transforms
    dataloader = SequenceDataLoader(dataset, BATCH_SIZE,
                                    transforms,
                                    shuffle_sequences=False,
                                    shuffle_frames=False,
                                    device=DEVICE)

    network = CenterNetDLA(n_channels=3, n_joints=22)
    network.load_state_dict(torch.load(MODEL_PATH))
    network = network.to(DEVICE)
    network.eval()
    with torch.no_grad():
        for sample in dataloader:
            sequences = sample['point_clouds']
            heat_maps, offsets, depths = network(sequences)
            skeletons = predict_3D_skeleton(heat_maps, offsets, depths,
                                            sample['intrinsic_matrix_inverted'],
                                            sample['rotation_matrix_inverted'],
                                            sample['translation_vector_inverted'],
                                            stride=4).detach().cpu().numpy()
            skeletons = np.moveaxis(skeletons, 1, -1)
            gt_skeletons = sample['skeletons'].detach().cpu().numpy()
            gt_skeletons = np.moveaxis(gt_skeletons, 1, -1)

            for skeleton, gt_skeleton in zip(skeletons, gt_skeletons):
                ax = plot_skeleton(skeleton)
                ax = plot_skeleton(gt_skeleton, ax)
                plt.show()
