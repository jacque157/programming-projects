import os
import torch
from tqdm import tqdm

from dataset.Dataset import StructuredCameraPointCloudDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Networks.CenterNet import CenterNet3DPoseEstimation, CenterNetOld
from visualisation.Visualisation import *

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
BATCH_SIZE = 32
BEST_MODEL_PATH = os.path.join('models',
                               'CenterNetPoseEstimationExperiment', #'CenterNetPoseEstimationExperimentOld', #'CenterNetPoseEstimationExperiment',
                               'net_45.pt')

def predict_3D_skeleton(heat_maps, offsets, depths,
                        intrinsic_matrix_inverted,
                        rotation_matrix_inverted,
                        translation_vector_inverted,
                        stride=4):
    
        b, j, h, w = heat_maps.shape
        heat_maps_flattened = torch.flatten(heat_maps, 2)
        idx = torch.argmax(heat_maps_flattened, 2)[:, :, None]
        xs = idx % w
        ys = idx // w
        positions_2D = torch.stack((xs, ys), 1).squeeze()

        offsets_flattened = torch.flatten(offsets, 2)
        x_offs = offsets_flattened[:, 0, None, :].expand(-1, j, -1)
        y_offs = offsets_flattened[:, 1, None, :].expand(-1, j, -1)
        relevant_x_offs = torch.take_along_dim(x_offs, idx, 2).squeeze()
        relevant_y_offs = torch.take_along_dim(y_offs, idx, 2).squeeze()
        relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 1)

        depths_flattened = -torch.flatten(depths, 2)
        relevant_depths = torch.take_along_dim(depths_flattened, idx, 2).squeeze()[:, None, :]

        positions_corrected = (positions_2D + relevant_offsets) * stride
        points = torch.cat((positions_corrected, torch.ones(b, 1, j, device=heat_maps.device)), 1)
        points *= relevant_depths
        points_3D = torch.bmm(rotation_matrix_inverted.expand(b, 3, 3),
                              translation_vector_inverted[None, :, None] + \
                              torch.bmm(intrinsic_matrix_inverted.expand(b, 3, 3), points))
        return points_3D

def MPJPE(prediction, ground_truth):
    return torch.mean(torch.sqrt(torch.sum((prediction - ground_truth) ** 2, 1)))

def rescale(pt_cloud, mean, min_, max_, a=-1, b=1):
    pt_cloud = (pt_cloud - a) * (max_[None, :, None] - min_[None, :, None]) / (b - a)
    pt_cloud += min_[None, :, None]
    pt_cloud += mean[None, :, None]
    return pt_cloud

if __name__ == '__main__':
    test_dataset = StructuredCameraPointCloudDataset(DATASET_PATH, 'test')
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

    network = CenterNet3DPoseEstimation(n_channels=3, n_joints=22)
    network.load_state_dict(torch.load(BEST_MODEL_PATH))
    network = network.to(DEVICE)
    network.eval()

    samples_count = 0
    accuracy_overall = 0
    with torch.no_grad():
        """for i, data in enumerate(test_dataloader):
            point_clouds = data['point_clouds']
            skeletons = data['skeletons']
            heat_maps, offsets, depths = network(point_clouds)
            predicted_3D_skeleton = predict_3D_skeleton(heat_maps, offsets, depths,
                                                        data['intrinsic_matrix_inverted'],
                                                        data['rotation_matrix_inverted'],
                                                        data['translation_vector_inverted'])
            
            accuracy_overall += MPJPE(predicted_3D_skeleton, skeletons)
            samples_count += 1

        acc = accuracy_overall / samples_count

        print(f'Testing MPJPE: {acc}')"""

        to_numpy = ToNumpy()
        for i, data in enumerate(test_dataloader):
            point_clouds = data['point_clouds']
            skeletons = data['skeletons']
            heat_maps, offsets, depths = network(point_clouds)
            predicted_3D_skeleton = predict_3D_skeleton(heat_maps, offsets, depths,
                                                        data['intrinsic_matrix_inverted'],
                                                        data['rotation_matrix_inverted'],
                                                        data['translation_vector_inverted'])
            predicted_joints = np.moveaxis(predicted_3D_skeleton.detach().cpu().numpy(), 1, 2)
            heat_maps = np.moveaxis(heat_maps.detach().cpu().numpy(), 1, 3)
            depth_maps = np.moveaxis(depths.detach().cpu().numpy(), 1, 3)
            offset_maps = np.moveaxis(offsets.detach().cpu().numpy(), 1, 3)
            data = to_numpy(data)
            point_clouds = data['point_clouds']
            skeletons = data['skeletons']
            
            for j, (heat_map, depth_map, offset_map,
                    point_cloud, skeleton, prediction) in enumerate(zip(heat_maps, depth_maps, offset_maps,
                                                                        point_clouds, skeletons, predicted_joints)):
                plt.imshow(point_cloud)
                plt.show()
                
                for k in range(22):
                    h_gt = data['heat_maps'][j, :, :, k]
                    exp_ = np.exp(h_gt)
                    sum_ = np.sum(exp_)
                    plt.imshow(h_gt)
                    plt.show()
                    h = heat_map[:, :, k]
                    plt.imshow(h)
                    plt.show()

                plt.imshow(depth_map)
                plt.show()

                plt.imshow(offset_map[:, :, 0])
                plt.show()

                plt.imshow(offset_map[:, :, 1])
                plt.show()
                
                #ax = plot_point_cloud(point_cloud, None, 1)
                ax = plot_skeleton(skeleton, None)
                plt.show()

                #ax = plot_point_cloud(point_cloud, None, 1)
                ax = plot_skeleton(prediction, None)
                plt.show()
            break
