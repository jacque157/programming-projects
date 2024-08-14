import os
import torch

from dataset.Dataset import StructuredCameraPointCloudDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Networks.CenterNet import CenterNetDLA
from Trainers.CenterNetTrainer import MPJPE, predict_3D_skeleton, rescale

from visualisation import Visualisation
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
EXPERIMENT_NAME = 'CenterNetDLA34PoseEstimation'
BATCH_SIZE = 16
REDUCTION = 4
BEST_MODEL = 23#15


def farthest_point_sampling(points, n):
    # https://minibatchai.com/2021/08/07/FPS.html
    points_left = np.arange(len(points)) 
    sample_inds = np.zeros(n, dtype='int') 
    dists = np.ones_like(points_left) * float('inf') 
    selected = 0
    sample_inds[0] = points_left[selected]
    points_left = np.delete(points_left, selected) 

    for i in range(1, n):
        last_added = sample_inds[i-1]
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) 
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) 
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]
        points_left = np.delete(points_left, selected)

    return points[sample_inds]

def plot_dataset(dataset, rows, cols, n_points=1024, eps=1e-4):
    pt_clouds, skeletons = [], []
    
    for i in range(rows * cols):
        sequence = np.random.randint(0, dataset.number_of_sequences())
        views = np.array([dataset.load_structured_point_clouds(sequence + 1, j + 1) for j in range(4)])
        v, s, h, w, c = views.shape
        frame = s // 2
        views = np.reshape(views[:, frame], (-1, 3))
        xs, ys, zs  = views[:, 0], views[:, 1], views[:, 2]
        xs_mask = np.logical_and(-eps < xs, xs < eps )
        ys_mask = np.logical_and(-eps < ys, ys < eps )
        zs_mask = np.logical_and(-eps < zs, zs < eps )
        mask = np.logical_not(np.logical_and(np.logical_and(xs_mask, ys_mask), zs_mask))
        points = views[mask]
        pt_cloud = farthest_point_sampling(points, n_points)
        pt_clouds.append(pt_cloud)
        skeleton = dataset.load_skeletons(sequence + 1)[frame]
        skeletons.append(skeleton)

    Visualisation.plot_dataset_examples(pt_clouds, skeletons, cols)

def animate_predictions(test_dataset, transforms, network, n_points=1024, eps=1e-4):
    with torch.no_grad():
        """max_ = 0
        hehe = 0
        for i in range(0, len(test_dataset), 4):
            pt = test_dataset[i]['point_clouds']
            ##print(i, len(pt))
            if len(pt) > max_:
                max_ = len(pt)
                hehe = i
        print(hehe)"""
        index = 0# 908 #np.random.randint(0, len(test_dataset))
        predicted_skeletons = []
        data = transforms(test_dataset[index])
        predicted_skeletons = []
        
        for i in range(0, len(data['point_clouds']), 32):       
            batch = data['point_clouds'][i : i + 32].to(DEVICE)
            heat_maps, offsets, depths = network(batch)
            predicted_3D_skeleton = predict_3D_skeleton(heat_maps, offsets, depths,
                                                        data['intrinsic_matrix_inverted'].to(DEVICE),
                                                        data['rotation_matrix_inverted'].to(DEVICE),
                                                        data['translation_vector_inverted'].to(DEVICE), REDUCTION)

            predicted_3D_skeleton = predicted_3D_skeleton.detach().cpu()
            predicted_3D_skeleton = torch.moveaxis(predicted_3D_skeleton, 1, 2).numpy()
            predicted_skeletons.extend(predicted_3D_skeleton)
            
        sequence_number = test_dataset.indexes[index // 4] + 1
        point_clouds = data['point_clouds']
        s, h, w, c = point_clouds.shape
        print(f'{s} frames')
        gt_skeletons = data['skeletons']
        sequence_start = index - (index % 4)
        pt_clouds = []
        views = np.array([dataset.load_structured_point_clouds(sequence_number, j + 1) for j in range(4)])
        v, s, h, w, c = views.shape
        #frame = s // 2
        #views = np.reshape(views[:, frame], (-1, 3))
        for frame in range(s):
            view = views[:, frame]
            points = np.reshape(view, (-1, 3))
            
            xs, ys, zs  = points[:, 0], points[:, 1], points[:, 2]
            xs_mask = np.logical_and(-eps < xs, xs < eps )
            ys_mask = np.logical_and(-eps < ys, ys < eps )
            zs_mask = np.logical_and(-eps < zs, zs < eps )
            mask = np.logical_not(np.logical_and(np.logical_and(xs_mask, ys_mask), zs_mask))
            points = points[mask]
            pt_cloud = farthest_point_sampling(points, n_points)
            pt_clouds.append(pt_cloud)

        gt_skeletons = gt_skeletons.detach().cpu()
        gt_skeletons = torch.moveaxis(gt_skeletons, 1, 2).numpy()
        pt_clouds = np.array(pt_clouds)

        Visualisation.animate_predictions(pt_clouds, gt_skeletons, predicted_skeletons)

        
if __name__ == '__main__':
    dataset = StructuredCameraPointCloudDataset(DATASET_PATH, 'all')
    test_dataset = StructuredCameraPointCloudDataset(DATASET_PATH, 'test')
    min_ = test_dataset.min(centered=True)
    max_ = test_dataset.max(centered=True)
    avg = test_dataset.avg()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=False),
                          Rescale(min_, max_, -1, 1, transform_skeletons=False),
                          ToTensor()])
    dataloader = SequenceDataLoader(dataset, BATCH_SIZE,
                                          transforms,
                                          shuffle_sequences=False,
                                          shuffle_frames=False,
                                          device=DEVICE)
    network = CenterNetDLA(n_channels=3, n_joints=22)
    network = network.to(DEVICE)
    path = os.path.join('models', f'{EXPERIMENT_NAME}', f'net_{BEST_MODEL}.pt')
    network.load_state_dict(torch.load(path))
    network.eval()

    #plot_dataset(dataset, 4, 6, n_points=1024)
    animate_predictions(test_dataset, transforms, network, n_points=1024)
        
    

        
        
        
