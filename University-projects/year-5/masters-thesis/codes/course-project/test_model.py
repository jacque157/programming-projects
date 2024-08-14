import os
import torch

from dataset.Dataset import PointCloudDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Networks.PointNet import PointNetPoseRegression
from visualisation.Visualisation import *

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
BATCH_SIZE = 32
BEST_MODEL_PATH = os.path.join('models',
                               'PointNetRegression_Points(2048)_Noise(5)_Rot(0, 0, 0)',
                               'net_40.pt')

def MPJPE(predictions, data):
    if 'center' in data:
        mean = data['center']
    else:
        device = a.device
        mean = torch.zeros(3, device=device)
                
    if 'a' in data:
        a = data['a']
        b = data['b']
        min_ = data['min_']
        max_ = data['max_']
        predictions_scaled = rescale(predictions, mean, min_, max_, a, b)
        skeletons_scaled = rescale(data['skeletons'], mean, min_, max_, a, b)
    else:
        predictions_scaled = predictions + mean[None, :, None]
        skeletons_scaled = data['skeletons'] + mean[None, :, None]
    return torch.mean(torch.sqrt(torch.sum(torch.square(predictions_scaled - skeletons_scaled), axis=1)))

def rescale(pt_cloud, mean, min_, max_, a=-1, b=1):
    pt_cloud = (pt_cloud - a) * (max_[None, :, None] - min_[None, :, None]) / (b - a)
    pt_cloud += min_[None, :, None]
    pt_cloud += mean[None, :, None]
    return pt_cloud

if __name__ == '__main__':
    test_dataset = PointCloudDataset(DATASET_PATH, 'test')
    min_ = test_dataset.minimum_point()
    max_ = test_dataset.maximum_point()
    avg = test_dataset.average_point()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=True),
                          Rescale(min_, max_, -1, 1, transform_skeletons=True),
                          ToTensor()])
    test_dataloader = SequenceDataLoader(test_dataset, BATCH_SIZE,
                                         transforms,
                                         shuffle_sequences=False,
                                         shuffle_frames=False,
                                         device=DEVICE)

    network = PointNetPoseRegression(n_channels=3, n_joints=22, device=DEVICE)
    network.load_state_dict(torch.load(BEST_MODEL_PATH))
    network = network.to(DEVICE)
    network.eval()

    samples_count = 0
    accuracy_overall = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            point_clouds = data['point_clouds']
            skeletons = data['skeletons']
            m3x3, m64x64, predicted_joints = network(point_clouds)
            accuracy_overall += MPJPE(predicted_joints, data)
            samples_count += 1

        acc = accuracy_overall / samples_count

        print(f'Testing MPJPE: {acc}')

    to_numpy = ToNumpy()
    for i, data in enumerate(test_dataloader):
        m3x3, m64x64, predicted_joints = network(point_clouds)
        predicted_joints = np.moveaxis(predicted_joints.detach().cpu().numpy(), 1, 2)
        data = to_numpy(data)
        point_clouds = data['point_clouds']
        skeletons = data['skeletons']
        
        for j, (point_cloud, skeleton, prediction) in enumerate(zip(point_clouds, skeletons, predicted_joints)):
            ax = plot_point_cloud(point_cloud, None, 1)
            ax.set_axis_off()
            plot_skeleton(skeleton, ax)
            plt.show()

            ax = plot_point_cloud(point_cloud, None, 1)
            ax.set_axis_off()
            plot_skeleton(prediction, ax)
            plt.show()
        break
        
