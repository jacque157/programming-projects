import os
import numpy as np
import matplotlib.pyplot as plt
from utils import *

path = 'Dataset/CMU/male/sequence_1/camera_1/poses.npz'
path_skeleton = 'Dataset/CMU/male/sequence_1/skeletons/skeletons.npz'

poses = np.load(path)
skeletons = np.load(path_skeleton)

for i in range(len(poses)):
    skeleton = skeletons[f'pose_{i}_skeleton']
    for cam in range(1, 5):
        path = f'Dataset/CMU/male/sequence_1/camera_{cam}/poses.npz'
        poses = np.load(path)
        pose = poses[f'pose_{i}']
        
        ax = plot_body(pose)
        plot_skeleton(skeleton, ax)
        plt.show()
