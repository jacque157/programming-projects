import os
import json
import numpy as np


class BaseDataset:
    def __init__(self, path, subset, transforms=None, protocol=(0, 1)):
        self.path = path
        self.subset = subset
        self.transforms = transforms
        self.protocol = protocol
        self.meta_data = self.load_metadata()

        indexes = self.get_indexes()
        k, l = protocol
        val_indexes = indexes[k::5]
        test_indexes = indexes[l::5]
        train_indexes = list((set(indexes) - set(val_indexes)) - set(test_indexes))
        train_indexes.sort()

        if subset in ('tr', 'trn', 'train', 'training'):
            self.indexes = train_indexes
        elif subset in ('vl', 'val', 'validation'):
            self.indexes = val_indexes
        elif subset in ('ts', 'tst', 'test', 'testing'):
            self.indexes = test_indexes
        elif subset == 'all':
            self.indexes = indexes
            
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        ...

    def get_indexes(self):
        ...

    def load_skeletons(self, sequence_number):
        skeletons_path = os.path.join(self.path, f'sequence_{sequence_number}', 'skeletons.npz')
        skeletons_archive = np.load(skeletons_path)
        skeletons = [skeletons_archive[f'skeleton_{i}'] for i in range(len(skeletons_archive))]
        return np.array(skeletons)

    def load_point_clouds(self, sequence_number):
        point_clouds_path = os.path.join(self.path, f'sequence_{sequence_number}', 'point_clouds.npz')
        point_clouds_archive = np.load(point_clouds_path)
        point_clouds = [point_clouds_archive[f'point_cloud_{i}'] for i in range(len(point_clouds_archive))]
        return np.array(point_clouds)

    def load_structured_point_clouds(self, sequence_number, camera_number):
        structured_point_clouds_path = os.path.join(self.path, f'sequence_{sequence_number}',
                                                    f'camera_{camera_number}', 'scans.npz')
        structured_point_clouds_archive = np.load(structured_point_clouds_path)
        structured_point_clouds = [structured_point_clouds_archive[f'frame_{i}'] for i in range(len(structured_point_clouds_archive))]
        return np.array(structured_point_clouds)

    def load_heatmaps(self, sequence_number, camera_number):
        heatmaps_path = os.path.join(self.path, f'sequence_{sequence_number}',
                                     f'camera_{camera_number}', 'heat_maps.npz')
        heatmaps = np.load(heatmaps_path)
        heatmaps = [heatmaps[f'heat_maps_{i}'] for i in range(len(heatmaps))]
        heatmaps = np.array(heatmaps)
        #exp_ = np.exp(heatmaps)
        #sum_ = np.sum(exp_, (1, 2))
        #heatmaps = exp_ / sum_[:, None, None, :]
        return heatmaps 

    def load_skeleton_offsets(self, sequence_number, camera_number):
        maps_path = os.path.join(self.path, f'sequence_{sequence_number}',
                                 f'camera_{camera_number}', 'skeletons_offset_maps.npz')
        maps = np.load(maps_path)
        maps = [maps[f'offset_maps_{i}'] for i in range(len(maps))]
        return np.array(maps)

    def load_skeleton_depths(self, sequence_number, camera_number):
        maps_path = os.path.join(self.path, f'sequence_{sequence_number}',
                                 f'camera_{camera_number}', 'skeletons_depth_maps.npz')
        maps = np.load(maps_path)
        maps = [maps[f'depth_maps_{i}'] for i in range(len(maps))]
        return np.abs(np.array(maps))
        
    def load_metadata(self):
        meta_data_path = os.path.join(self.path, 'meta_data.json')
        with open(meta_data_path) as file:
            meta_data = json.load(file)

        for key, value in meta_data.items():
            meta_data[key] = np.array(value)

        meta_data["intrinsic_matrix_inverted"] = np.linalg.inv(meta_data["intrinsic_matrix"][0 : 3, 0 : 3])
        
        rots_inv = np.array([extrinsic_matrix[0:3, 0:3].T for extrinsic_matrix in meta_data["extrinsic_matrices"]])
        meta_data["rotations_inverted"] = rots_inv
        meta_data["translation_inverted"] = -np.array([extrinsic_matrix[0:3, 3] for extrinsic_matrix in meta_data["extrinsic_matrices"]])
        return meta_data

    def load_segmentation(self, sequence_number, camera_number):
        segmentations_path = os.path.join(self.path, f'sequence_{sequence_number}',
                                          f'camera_{camera_number}', 'segmentations.npz')
        segmentations = np.load(segmentations_path)
        segmentations = [segmentations[f'segmentation_{i}'] for i in range(len(segmentations))]
        return np.abs(np.array(segmentations))
        
    def intrinsic_matrix(self):
        return self.meta_data["intrinsic_matrix"]

    def extrinsic_matrix(self, camera):
        return self.meta_data["extrinsic_matrices"][camera - 1]

    def number_of_sequences(self):
        return self.meta_data["sequences"]

    def name(self):
        return self.meta_data["name"]

    def average_pixel(self):
        return self.meta_data["average_pixel"]

    def average_point(self):
        return self.meta_data["average_point"]

    def average_camera_point(self):
        return self.meta_data["average_camera_point"]

    def number_of_frames(self):
        return self.meta_data["frames"]

    def minimum_point(self):
        return self.meta_data["min_point"]

    def maximum_point(self):
        return self.meta_data["max_point"]

    def minimum_camera_pixel(self):
        return self.meta_data["min_camera_point"]

    def maximum_camera_pixel(self):
        return self.meta_data["max_camera_point"]

    def minimum_world_pixel(self):
        return self.meta_data["min_pixel"]

    def maximum_world_pixel(self):
        return self.meta_data["max_pixel"]

    def intrinsic_matrix_inverted(self):
        return self.meta_data["intrinsic_matrix_inverted"]

    def rotation_matrix_inverted(self, camera):
        return self.meta_data["rotations_inverted"][camera - 1]

    def translation_vector_inverted(self, camera):
        return self.meta_data["translation_inverted"][camera - 1]


class PointCloudDataset(BaseDataset):
    def __init__(self, path, subset, transforms=None, protocol=(0, 1)):
        super().__init__(path, subset, transforms, protocol)

    def __getitem__(self, index):
        sequence_number = self.indexes[index] + 1
        data = {}

        data['point_clouds'] = self.load_point_clouds(sequence_number)
        data['skeletons'] = self.load_skeletons(sequence_number)
        data['name'] = self.name()

        if self.transforms:
            return self.transforms(data)
        return data

    def get_indexes(self):
        return list(range(self.number_of_sequences()))

class StructuredWorldPointCloudDataset(BaseDataset):
    def __init__(self, path, subset, transforms=None, protocol=(0, 1)):
        super().__init__(path, subset, transforms, protocol)

    def __getitem__(self, index):
        sequence_number = self.indexes[index // 4] + 1
        camera_number = (index % 4) + 1
        data = {}

        data['point_clouds'] = self.load_structured_point_clouds(sequence_number, camera_number)
        data['skeletons'] = self.load_skeletons(sequence_number)
        data['name'] = self.name()

        if self.transforms:
            return self.transforms(data)
        return data

    def get_indexes(self):
        return list(range(self.number_of_sequences()))
    
    def __len__(self):
        return len(self.indexes) * 4

class StructuredCameraPointCloudDataset(StructuredWorldPointCloudDataset):
    def __init__(self, path, subset, transforms=None, protocol=(0, 1)):
        super().__init__(path, subset, transforms, protocol)

    def __getitem__(self, index):
        sequence_number = self.indexes[index // 4] + 1
        camera_number = (index % 4) + 1
        data = {}

        point_clouds = self.load_structured_point_clouds(sequence_number, camera_number)
        s, h, w, x = point_clouds.shape
        x = point_clouds[:, :, :, 0]
        y = point_clouds[:, :, :, 1]
        z = point_clouds[:, :, :, 2]
        mask = np.logical_not(np.logical_and(np.logical_and(x == 0, y == 0), z == 0))
        points = point_clouds[mask]
        points = np.concatenate((points, np.ones((len(points), 1))), 1)
        camera_space_points = (self.extrinsic_matrix(camera_number) @ points.T).T
        point_clouds[mask] = camera_space_points[:, : 3]
        skeletons_3D = self.load_skeletons(sequence_number)
        heatmaps = self.load_heatmaps(sequence_number, camera_number)  
        data['point_clouds'] = point_clouds
        data['skeletons'] = skeletons_3D#[:, 0 : 22, :]
        data['heat_maps'] = heatmaps#[:, :, :, 0 : 22]
        data['name'] = self.name()
        data['intrinsic_matrix_inverted'] = self.intrinsic_matrix_inverted()
        data['rotation_matrix_inverted'] = self.rotation_matrix_inverted(camera_number) #np.broadcast_to(self.rotation_matrix_inverted(camera_number), (s, 3, 3))
        data['translation_vector_inverted'] = self.translation_vector_inverted(camera_number)
        """data['segmentations'] = self.load_segmentation(sequence_number, camera_number)   

        s, n, c = skeletons_3D.shape
        skeletons_3D_flattened = np.reshape(skeletons_3D, (s * n, c))
        skeletons_3D_flattened = np.concatenate((skeletons_3D_flattened, np.ones((s * n, 1))), 1)
        skeletons_2D_flattened = (self.intrinsic_matrix() @ self.extrinsic_matrix(camera_number) @ skeletons_3D_flattened.T).T
        depths_flattened = skeletons_2D_flattened[:, -1]
        skeletons_2D_flattened = skeletons_2D_flattened / depths_flattened[:, None]
        depths = np.reshape(depths_flattened, (s, n))
        skeletons_2D = np.reshape(skeletons_2D_flattened[:, :2], (s, n, 2))
        data['skeletons_2D'] = skeletons_2D
        data['skeletons_depths'] = np.abs(depths)[:, :, None]"""
        data['2D_skeletons_depths'] = self.load_skeleton_depths(sequence_number, camera_number) 
        data['2D_skeletons_offsets'] = self.load_skeleton_offsets(sequence_number, camera_number) 

        if self.transforms:
            return self.transforms(data)
        return data

    def __len__(self):
        return len(self.indexes) * 4
        #return 4
