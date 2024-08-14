import os
from typing import Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time


from Transforms import *
from Transforms import ToTensor
from utils import *


class Poses3D(Dataset):
    def __init__(self, path, name, subset=None, transform=ToTensor(), protocol=(0, 1), include_segmentation=True):
        self.path = path
        self.name = name
        self.transform = transform
        self.include_segmentation = include_segmentation
        self.joints = 22
        n = self.get_size()
        indexes = set(range(n))
        k, l = protocol
        validation_indexes = set(range(k % 5, n, 5)) # k, k + 5, k + 10, ...
        testing_indexes = set(range(l % 5, n, 5)) # l, l + 5, l + 10, ...
        training_indexes = indexes - validation_indexes - testing_indexes
        if subset in ('validation', 'val', 'vl'):
            self.indexes = self.create_indexes(validation_indexes)
        elif subset in ('testing', 'test', 'tst', 'ts'):
            self.indexes = self.create_indexes(testing_indexes)
        elif subset in ('training', 'train', 'trn', 'tr'):
            self.indexes = self.create_indexes(training_indexes)
        else:
            self.indexes = self.create_indexes(indexes)
        self.indexes = list(self.indexes)

    def create_indexes(self, sequences):
        indexes = []
        for seq_number in sequences:
            low = seq_number * 4
            high = low + 4
            indexes.extend(range(low, high))
        return indexes

    def get_size(self):
        n = 0
        path = os.path.join(self.path, self.name, 'male')
        if os.path.exists(path):
            n += len(os.listdir(path))

        path = os.path.join(self.path, self.name, 'female')
        if os.path.exists(path):
            n += len(os.listdir(path))

        """number_of_cameras = 4
        return n * number_of_cameras"""
        return n 

    def __getitem__(self, index) -> Union[dict[str, np.array], dict[str, torch.Tensor]]:
        index = self.indexes[index]
        camera_index = (index % 4) + 1
        sequence = (index // 4) + 1

        """frames = load_number_of_frames(self.path, self.name, sequence)[0]
        if frames is None:
            return None, None, None, None"""
        
        poses = load_poses(self.path, self.name, sequence, camera_index) #[load_body(self.path, self.name, sequence, frame, camera_index) for frame in range(frames)]
        poses = [poses[f'pose_{frame}'] for frame in range(len(poses))]
        skeletons = load_skeletons(self.path, self.name, sequence) #[load_skeleton(self.path, self.name, sequence, frame) for frame in range(frames)]
        skeletons = [skeletons[f'pose_{frame}_skeleton'] for frame in range(len(skeletons))]
        if self.include_segmentation:
            segmentations = load_segmentations(self.path, self.name, sequence, camera_index) #[load_segmentation(self.path, self.name, sequence, frame, camera_index) for frame in range(frames)]
            segmentations = [segmentations[f'pose_{frame}_segmentation'] for frame in range(len(segmentations))]
            segmentations = np.array(segmentations, dtype=segmentations[0].dtype)
        poses = np.array(poses, dtype=poses[0].dtype)
        skeletons = np.array(skeletons, dtype=skeletons[0].dtype)
        

        x = poses[:, :, :, 0]
        y = poses[:, :, :, 1]
        z = poses[:, :, :, 2]
        mask = np.logical_not(np.logical_and(np.logical_and(x == 0, y == 0), z == 0))
        centres = skeletons[:, 0, :]

        sample = {'sequences' : poses, 'valid_points' : mask, 'key_points' : skeletons, 'root_key_points' : centres}
        if self.include_segmentation:
            sample['segmentations'] = segmentations
        if self.transform is not None:
            sample = self.transform(sample)
        return sample # structured point clouds in sequence, binary masks for valid values (invalid point: (0, 0, 0)), points of skeletons, positions  of root point (pelvis) in skeletons 

    def __len__(self):
        return len(self.indexes)

class DummyPose3D(Poses3D):
    def __init__(self, path, name, subset=None, transform=ToTensor(), size=1):
        self.size = size
        super().__init__(path, name, subset, transform, protocol=(0, 1), include_segmentation=True)
        
    def get_size(self):
        return self.size
    

class TwoFrames(Dataset):
    def __init__(self, path, name, subset=None, transform=ToTensor(), protocol=(0, 1), include_segmentation=True):
        self.path = path
        self.name = name
        self.transform = transform
        self.include_segmentation = include_segmentation
        self.joints = 22
        self.indexes = [0]

    def get_size(self):
        return 1

    def __getitem__(self, index) -> Union[dict[str, np.array], dict[str, torch.Tensor]]:
        index = self.indexes[index]
        camera_index = (index % 4) + 1
        sequence = (index // 4) + 1

        """frames = load_number_of_frames(self.path, self.name, sequence)[0]
        if frames is None:
            return None, None, None, None"""
        
        poses = load_poses(self.path, self.name, sequence, camera_index) 
        poses = [poses[f'pose_{frame}'] for frame in range(2)]
        skeletons = load_skeletons(self.path, self.name, sequence) 
        skeletons = [skeletons[f'pose_{frame}_skeleton'] for frame in range(2)]
        if self.include_segmentation:
            segmentations = load_segmentations(self.path, self.name, sequence, camera_index) 
            segmentations = [segmentations[f'pose_{frame}_segmentation'] for frame in range(2)]
            segmentations = np.array(segmentations, dtype=segmentations[0].dtype)
        poses = np.array(poses, dtype=poses[0].dtype)
        skeletons = np.array(skeletons, dtype=skeletons[0].dtype)
        

        x = poses[:, :, :, 0]
        y = poses[:, :, :, 1]
        z = poses[:, :, :, 2]
        mask = np.logical_not(np.logical_and(np.logical_and(x == 0, y == 0), z == 0))
        centres = skeletons[:, 0, :]

        sample = {'sequences' : poses, 'valid_points' : mask, 'key_points' : skeletons, 'root_key_points' : centres}
        if self.include_segmentation:
            sample['segmentations'] = segmentations
        if self.transform:
            sample = self.transform(sample)
        return sample # structured point clouds in sequence, binary masks for valid values (invalid point: (0, 0, 0)), points of skeletons, positions  of root point (pelvis) in skeletons 

    def __len__(self):
        return len(self.indexes)



if __name__ == '__main__':
    DATASET_PATH = os.path.join('..', 'codes', 'Dataset')
    SEGMENTATION = False
    transform = None
    
    ACCAD_tr_dataset = Poses3D(DATASET_PATH, 
                               'ACCAD', 
                               subset='training', 
                               transform=transform,
                               protocol=(0, 1), 
                               include_segmentation=SEGMENTATION)
    
    ACCAD_val_dataset = Poses3D(DATASET_PATH, 
                                'ACCAD', 
                                subset='validation', 
                                transform=transform,
                                protocol=(0, 1), 
                                include_segmentation=SEGMENTATION)
    
    CMU_tr_dataset = Poses3D(DATASET_PATH,
                             'CMU',
                             subset='training',
                             transform=transform,
                             protocol=(0, 1), 
                             include_segmentation=SEGMENTATION)
    
    CMU_val_dataset = Poses3D(DATASET_PATH,
                              'CMU',
                              subset='validation',
                              transform=transform,
                              protocol=(0, 1),
                              include_segmentation=SEGMENTATION)
    
    EKUT_tr_dataset = Poses3D(DATASET_PATH,
                              'EKUT',
                              subset='training',
                              transform=transform,
                              protocol=(0, 1), 
                              include_segmentation=SEGMENTATION)
    
    EKUT_val_dataset = Poses3D(DATASET_PATH,
                               'EKUT',
                               subset='validation',
                               transform=transform,
                               protocol=(0, 1),
                               include_segmentation=SEGMENTATION)
    
    Eyes_Japan_tr_dataset = Poses3D(DATASET_PATH,
                                    'Eyes_Japan',
                                    subset='training',
                                    transform=transform,
                                    protocol=(0, 1),
                                    include_segmentation=SEGMENTATION)
    
    Eyes_Japan_val_dataset = Poses3D(DATASET_PATH,
                                     'Eyes_Japan',
                                     subset='validation',
                                     transform=transform,
                                     protocol=(0, 1),
                                     include_segmentation=SEGMENTATION)
    print(os.path.exists(DATASET_PATH))
    print(len(ACCAD_tr_dataset) + len(CMU_tr_dataset) + len (EKUT_tr_dataset) + len(Eyes_Japan_tr_dataset))
    print(len(ACCAD_val_dataset) + len(CMU_val_dataset) + len (EKUT_val_dataset) + len(Eyes_Japan_val_dataset))
    #print(len(ACCAD_ts_dataset))
