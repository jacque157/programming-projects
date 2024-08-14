import torch
import os
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tools import Tools
from transforms import ToTensor
from visualisation import Sketch


class KinopticDataset(Dataset):

    def __init__(self, path, transform=None):
        self.last_file_index = None
        self.cloud = None
        self.pose = None
        self.path = path
        self.size = self.dataset_size()
        self.transform = transform

    def dataset_size(self):
        poses_path = f"{self.path}/annotations"
        clouds_path = f"{self.path}/point_clouds"
        size1, size2 = len(os.listdir(poses_path)), len(os.listdir(clouds_path))
        return size1 if size1 < size2 else size2

    def __len__(self):
        return self.size

    def __getitem__(self, index) -> T_co:
        file_index = index + 1

        if file_index == self.last_file_index:
            cloud = self.cloud
            pose = self.pose
        else:
            self.last_file_index = file_index
            poses_path = f"{self.path}/annotations/pose{file_index}.npy"
            clouds_path = f"{self.path}/point_clouds/point_cloud{file_index}.npy"
            cloud = Tools.np_file_to_array(clouds_path)
            pose = Tools.np_file_to_array(poses_path)
            if self.transform:
                cloud = self.transform(cloud)
                pose = self.transform(pose)
            self.cloud = cloud
            self.pose = pose

        return {"point_cloud": cloud, "annotations": pose}

    def show(self, index):
        cloud, pose = self[index]["point_cloud"], self[index]["annotations"]
        if torch.is_tensor(cloud):
            cloud = cloud.numpy()
        cloud = cloud.transpose(1, 0)
        if torch.is_tensor(pose):
            pose = pose.numpy()
        Sketch.plot_annotated_ptcloud(cloud, pose)


def main():
    dataset = KinopticDataset("data")
    i = len(dataset) - 300
    print(i)
    dataset.show(i)
    dataset_t = KinopticDataset("data", ToTensor())
    cloud_t, pose_t = dataset_t[0].values()
    cloud, pose = dataset[0].values()
    print(cloud_t.shape, pose_t.shape)
    print(cloud.shape, pose.shape)
    dataset_t.show(0)


if __name__ == '__main__':
    main()
    
