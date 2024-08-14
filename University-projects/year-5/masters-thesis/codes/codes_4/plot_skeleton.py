import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset.Dataset import StructuredCameraPointCloudDataset
from visualisation import Visualisation


DATASET_PATH = os.path.join('dataset', 'CMU')


def structured2unstructured(structured_point_cloud, eps=1e-4):
    xs, ys, zs  = structured_point_cloud[:, :, 0], structured_point_cloud[:, :, 1], structured_point_cloud[:, :, 2]
    xs_mask = np.logical_and(-eps < xs, xs < eps )
    ys_mask = np.logical_and(-eps < ys, ys < eps )
    zs_mask = np.logical_and(-eps < zs, zs < eps )
    mask = np.logical_not(np.logical_and(np.logical_and(xs_mask, ys_mask), zs_mask))
    return structured_point_cloud[mask]

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

if __name__ == '__main__':
    test_dataset = StructuredCameraPointCloudDataset(DATASET_PATH, 'all')
    frame = 0
    sequence = 1
    points = np.concatenate([structured2unstructured(test_dataset.load_structured_point_clouds(sequence, i)[frame]) for i in range(1, 5)], 0)
    point_cloud = farthest_point_sampling(points, 4096)
    skeleton = test_dataset.load_skeletons(sequence)[frame]
    Visualisation.plot_example(point_cloud, target_skeleton=skeleton, plot_labels=True)
