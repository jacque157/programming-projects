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

    def load_images(self, sequence_number, camera_number):
        images_path = os.path.join(self.path, f'sequence_{sequence_number}',
                                   f'camera_{camera_number}', 'imgs.npz')
        images_archive = np.load(images_path)
        images = [images_archive[f'frame_{i}'] for i in range(len(images_archive))]
        return np.array(images, dtype=np.float64)

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

    def load_visibility_mask(self, sequence_number, camera_number):
        mask_path = os.path.join(self.path, f'sequence_{sequence_number}',
                                 f'camera_{camera_number}', 'visible_joints.npz')
        mask = np.load(mask_path)
        mask = [mask[f'frame_{i}'] for i in range(len(mask))]
        return np.array(mask)
        
    def intrinsic_matrix(self):
        return self.meta_data["intrinsic_matrix"]

    def extrinsic_matrix(self, camera):
        return self.meta_data["extrinsic_matrices"][camera - 1]

    def camera_position(self, camera):
        return self.meta_data["camera_positions"][camera - 1]

    def number_of_sequences(self):
        return self.meta_data["sequences"]

    def name(self):
        return self.meta_data["name"]

    def average_pixel(self):
        return self.meta_data["avg_pixel"]

    def average_point(self):
        return self.meta_data["avg_point"]

    def average_camera_point(self):
        return self.meta_data["avg_camera_point"]

    def std_pixel(self):
        return self.meta_data["std_pixel"]

    def std_point(self):
        return self.meta_data["std_point"]

    def std_camera_point(self):
        return self.meta_data["std_camera_point"]
    
    def number_of_frames(self):
        return self.meta_data["frames"]

    def minimum_point(self, centered=True):
        return self.meta_data["min_point_corrected"] if centered else self.meta_data["min_point"]

    def maximum_point(self, centered=True):
        return self.meta_data["max_point_corrected"]  if centered else self.meta_data["max_point"] 

    def minimum_camera_point(self, centered=True):
        return self.meta_data["min_camera_point_corrected"] if centered else self.meta_data["min_camera_point"]

    def maximum_camera_point(self, centered=True):
        return self.meta_data["max_camera_point_corrected"] if centered else self.meta_data["max_camera_point"]

    def minimum_pixel(self, centered=True):
        return self.meta_data["min_pixel_corrected"] if centered else self.meta_data["min_pixel"]

    def maximum_pixel(self, centered=True):
        return self.meta_data["max_pixel_corrected"] if centered else self.meta_data["max_pixel"]

    def intrinsic_matrix_inverted(self):
        return self.meta_data["intrinsic_matrix_inverted"]

    def rotation_matrix_inverted(self, camera):
        return self.meta_data["rotations_inverted"][camera - 1]

    def translation_vector_inverted(self, camera):
        return self.meta_data["translation_inverted"][camera - 1]


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

    def avg(self):
        return self.average_point()

    def std(self):
        return self.std_point()

    def min(self, centered):
        return self.minimum_point(centered)

    def max(self, centered):
        return self.maximum_point(centered)

class StructuredCameraPointCloudDataset(StructuredWorldPointCloudDataset):
    def __init__(self, path, subset, transforms=None, protocol=(0, 1), camera_space_skeletons=False):
        super().__init__(path, subset, transforms, protocol)
        self.camera_space_skeletons = camera_space_skeletons

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

        skeletons = self.load_skeletons(sequence_number)
        if self.camera_space_skeletons:
            s, j, c = skeletons.shape
            assert c == 3
            points = np.reshape(skeletons, (s * j, c))
            points = np.concatenate((points, np.ones((len(points), 1))), 1)
            camera_space_joints = (self.extrinsic_matrix(camera_number) @ points.T).T
            camera_space_joints = camera_space_joints[:, :3]
            skeletons = np.reshape(camera_space_joints, (s, j, c))

            data['point_clouds'] = point_clouds
            data['skeletons'] = skeletons 
            data['name'] = self.name()
            #data['intrinsic_matrix'] = self.intrinsic_matrix()
            #data['extrinsic_matrix'] = self.extrinsic_matrix(camera_number)
            data['rotation_matrix_inverted'] = self.rotation_matrix_inverted(camera_number) 
            data['translation_vector_inverted'] = self.translation_vector_inverted(camera_number)
        else: 
            data['point_clouds'] = point_clouds
            data['skeletons'] = skeletons
            data['heat_maps'] = self.load_heatmaps(sequence_number, camera_number)  
            data['name'] = self.name()
            #data['intrinsic_matrix'] = self.intrinsic_matrix()
            #data['extrinsic_matrix'] = self.extrinsic_matrix(camera_number)
            data['intrinsic_matrix_inverted'] = self.intrinsic_matrix_inverted()
            data['rotation_matrix_inverted'] = self.rotation_matrix_inverted(camera_number) 
            data['translation_vector_inverted'] = self.translation_vector_inverted(camera_number)
            data['2D_skeletons_depths'] = self.load_skeleton_depths(sequence_number, camera_number) 
            data['2D_skeletons_offsets'] = self.load_skeleton_offsets(sequence_number, camera_number) 

        if self.transforms:
            return self.transforms(data)
        return data

    def __len__(self):
        return len(self.indexes) * 4

    def avg(self):
        return self.average_camera_point()

    def std(self):
        return self.std_camera_point()

    def min(self, centered):
        return self.minimum_camera_point(centered)

    def max(self, centered):
        return self.maximum_camera_point(centered)

class StructuredCameraPointCloudDatasetFor2DPoseEstimation(StructuredCameraPointCloudDataset):
    def __init__(self, path, subset, transforms=None, protocol=(0, 1)):
        super().__init__(path, subset, transforms, protocol, False)
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

        data['point_clouds'] = point_clouds
        data['skeletons'] = self.load_skeletons(sequence_number)
        data['heat_maps'] = self.load_heatmaps(sequence_number, camera_number)  
        data['name'] = self.name()
        #data['intrinsic_matrix'] = self.intrinsic_matrix()
        #data['extrinsic_matrix'] = self.extrinsic_matrix(camera_number)
        data['intrinsic_matrix_inverted'] = self.intrinsic_matrix_inverted()
        data['rotation_matrix_inverted'] = self.rotation_matrix_inverted(camera_number) 
        data['translation_vector_inverted'] = self.translation_vector_inverted(camera_number)
            
        if self.transforms:
            return self.transforms(data)
        return data

class ImageDatasetFor2DPoseEstimation(StructuredCameraPointCloudDataset):
    def __init__(self, path, subset, transforms=None, protocol=(0, 1)):
        super().__init__(path, subset, transforms, protocol, False)
    def __getitem__(self, index):
        sequence_number = self.indexes[index // 4] + 1
        camera_number = (index % 4) + 1
        data = {}

        data['images'] = self.load_images(sequence_number, camera_number)
        data['skeletons'] = self.load_skeletons(sequence_number)
        data['heat_maps'] = self.load_heatmaps(sequence_number, camera_number)  
        data['name'] = self.name()
        #data['intrinsic_matrix'] = self.intrinsic_matrix()
        #data['extrinsic_matrix'] = self.extrinsic_matrix(camera_number)
        data['intrinsic_matrix_inverted'] = self.intrinsic_matrix_inverted()
        data['rotation_matrix_inverted'] = self.rotation_matrix_inverted(camera_number) 
        data['translation_vector_inverted'] = self.translation_vector_inverted(camera_number)
            
        if self.transforms:
            return self.transforms(data)
        return data

class ImageDataset(StructuredCameraPointCloudDataset):
    def __init__(self, path, subset, transforms=None, protocol=(0, 1), camera_space_skeletons=False):
        super().__init__(path, subset, transforms, protocol, camera_space_skeletons)

    def __getitem__(self, index):
        sequence_number = self.indexes[index // 4] + 1
        camera_number = (index % 4) + 1
        data = {}

        skeletons = self.load_skeletons(sequence_number)
        if self.camera_space_skeletons:
            s, j, c = skeletons.shape
            assert c == 3
            points = np.reshape(skeletons, (s * j, c))
            points = np.concatenate((points, np.ones((len(points), 1))), 1)
            camera_space_joints = (self.extrinsic_matrix(camera_number) @ points.T).T
            camera_space_joints = camera_space_joints[:, :3]
            skeletons = np.reshape(camera_space_joints, (s, j, c))

            data['images'] = self.load_images(sequence_number, camera_number)
            data['skeletons'] = skeletons 
            data['name'] = self.name()
            #data['intrinsic_matrix'] = self.intrinsic_matrix()
            #data['extrinsic_matrix'] = self.extrinsic_matrix(camera_number)
            data['rotation_matrix_inverted'] = self.rotation_matrix_inverted(camera_number) 
            data['translation_vector_inverted'] = self.translation_vector_inverted(camera_number)
        else:     
            data['images'] = self.load_images(sequence_number, camera_number)
            data['skeletons'] = skeletons
            data['heat_maps'] = self.load_heatmaps(sequence_number, camera_number)  
            data['name'] = self.name()
            #data['intrinsic_matrix'] = self.intrinsic_matrix()
            #data['extrinsic_matrix'] = self.extrinsic_matrix(camera_number)
            data['intrinsic_matrix_inverted'] = self.intrinsic_matrix_inverted()
            data['rotation_matrix_inverted'] = self.rotation_matrix_inverted(camera_number) 
            data['translation_vector_inverted'] = self.translation_vector_inverted(camera_number)
            data['2D_skeletons_depths'] = self.load_skeleton_depths(sequence_number, camera_number) 
            data['2D_skeletons_offsets'] = self.load_skeleton_offsets(sequence_number, camera_number) 

        if self.transforms:
            return self.transforms(data)
        return data

    def __len__(self):
        return len(self.indexes) * 4

    def avg(self):
        return self.average_pixel()

    def std(self):
        return self.std_pixel()

    def min(self, centered):
        return self.minimum_pixel(centered)

    def max(self, centered):
        return self.maximum_pixel(centered)

class MultiviewStructuredCameraPointCloudDataset(StructuredCameraPointCloudDataset):
    def __init__(self, path, subset, transforms=None, protocol=(0, 1)):
        super().__init__(path, subset, transforms, protocol, False)

    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, index):
        sequence_number = self.indexes[index] + 1
        
        multi_view_point_clouds = []
        for camera_number in range(1, 5):
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
            multi_view_point_clouds.append(point_clouds)
        multi_view_point_clouds = np.moveaxis(multi_view_point_clouds, 0, 1)
        
        skeletons = self.load_skeletons(sequence_number)
        rotation_matrices = np.array([self.extrinsic_matrix(camera_number)[0:3, 0:3]
                                      for camera_number in range(1, 5)])
        translations = np.array([self.extrinsic_matrix(camera_number)[0:3,3]
                                      for camera_number in range(1, 5)])
        camera_positions = np.array([self.camera_position(camera_number)
                                      for camera_number in range(1, 5)])
        visible_joints = np.array([self.load_visibility_mask(sequence_number, camera_number)
                                   for camera_number in range(1, 5)])
        heat_maps = np.array([self.load_heatmaps(sequence_number, camera_number)
                                   for camera_number in range(1, 5)])
        visible_joints = np.moveaxis(visible_joints, 0, 1)
        heat_maps = np.moveaxis(heat_maps, 0, 1)

        data = {}
        data['point_clouds'] = multi_view_point_clouds
        data['skeletons'] = skeletons
        data['heat_maps'] = heat_maps
        data['name'] = self.name()
        data['intrinsic_matrix'] = self.intrinsic_matrix()[:3, :3]#np.concatenate((self.intrinsic_matrix(), [[0, 0, 0, 1]]), 0)
        #print(rotation_matrices.shape)
        data['camera_rotations'] = rotation_matrices
        data['camera_translations'] = translations
        data['camera_positions'] = camera_positions
        data['visible_joints'] = visible_joints

        if self.transforms:
            return self.transforms(data)
        return data
    
