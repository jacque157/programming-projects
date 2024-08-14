import os
import numpy as np


def split_data(folder, output):
    poses_path = f"{folder}/annotations"
    clouds_path = f"{folder}/point_clouds"
    poses_output = f"{output}/annotations"
    clouds_output = f"{output}/point_clouds"
    i = 1
    for j in range(len(os.listdir(poses_path))):
        poses = np.load(f"{poses_path}/poses{j + 1}.npy")
        dim = poses.shape
        poses = poses.reshape((dim[0], dim[1]))
        for pose in poses:
            pose = pose.astype('int')
            np.save(f"{poses_output}/pose{i}.npy", pose)
            i += 1
    i = 1
    for j in range(len(os.listdir(clouds_path))):
        clouds = np.load(f"{clouds_path}/point_clouds{j + 1}.npy")
        for cloud in clouds:
            cloud = cloud.astype('float')
            cloud = np.transpose(cloud, (1, 0))
            np.save(f"{clouds_output}/point_cloud{i}.npy", cloud)
            i += 1


if __name__ == '__main__':
    split_data("data", "split")
