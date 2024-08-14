import numpy as np
import torch


"""class SequenceDataLoader:
    class __Iterator:
        def __init__(self, dataset, batch_size, transforms=None, shuffle_sequences=True, shuffle_frames=True, device='cpu'):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle_sequences = shuffle_sequences
            self.shuffle_frames = shuffle_frames
            self.device = device
            self.transforms = transforms
            
            self.data = None
            self.index = 0
            self.batch_start = 0
            
            if shuffle_sequences:
                self.indexes = np.random.permutation(len(dataset))
            else:
                self.indexes = np.arange(len(dataset))
            
        def __next__(self):
            if self.index >= len(self.indexes):
                raise StopIteration
            
            if self.data is None:
                self.batch_start = 0
                sequence_index = self.indexes[self.index]
                self.data = self.dataset[sequence_index]   
                if self.shuffle_frames:
                    data_indexes = np.random.permutation(len(self.data['point_clouds']))
                    self.data['point_clouds'] = self.data['point_clouds'][data_indexes]
                    self.data['skeletons'] = self.data['skeletons'][data_indexes]

                    if 'heat_maps' in self.data:
                        #self.data['2D_skeletons_depths'] = self.data['2D_skeletons_depths'][data_indexes]
                        #self.data['2D_skeletons_offsets'] = self.data['2D_skeletons_offsets'][data_indexes]
                        self.data['heat_maps'] = self.data['heat_maps'][data_indexes]
                        self.data['skeletons_2D'] = self.data['skeletons_2D'][data_indexes]
                        self.data['skeletons_depths'] = self.data['skeletons_depths'][data_indexes]
                        self.data['rotation_matrix_inverted'] = self.data['rotation_matrix_inverted'][data_indexes]
                    
            point_clouds = self.data['point_clouds']
            skeletons = self.data['skeletons']
                               
            batch_end = self.batch_start + self.batch_size
            point_clouds_batched = point_clouds[self.batch_start : batch_end]
            skeletons_batched = skeletons[self.batch_start : batch_end]

            if 'heat_maps' in self.data:
                heatmaps = self.data['heat_maps']
                #skeletons_offsets = self.data['2D_skeletons_offsets']
                #skeleton_depths = self.data['2D_skeletons_depths']
                skeletons_2D = self.data['skeletons_2D']
                skeletons_depths = self.data['skeletons_depths']
                rotation_matrix_inverted = self.data['rotation_matrix_inverted']
                heatmaps_batched = heatmaps[self.batch_start : batch_end]
                skeletons_2D_batched = skeletons_2D[self.batch_start : batch_end]
                skeletons_depths_batched = skeletons_depths[self.batch_start : batch_end]
                rotation_matrix_inverted_batched = rotation_matrix_inverted[self.batch_start : batch_end]
                #skeletons_offsets_batched = skeletons_offsets[self.batch_start : batch_end]
                #skeleton_depths_batched = skeleton_depths[self.batch_start : batch_end]
                
            while len(point_clouds_batched) < self.batch_size:
                self.index += 1
                if self.index >= len(self.indexes):
                    break
                sequence_index = self.indexes[self.index]
                self.data = self.dataset[sequence_index]
                if self.shuffle_frames:
                    data_indexes = np.random.permutation(len(self.data['point_clouds']))
                    self.data['point_clouds'] = self.data['point_clouds'][data_indexes]
                    self.data['skeletons'] = self.data['skeletons'][data_indexes]
                    
                    if 'heat_maps' in self.data:
                        #self.data['2D_skeletons_depths'] = self.data['2D_skeletons_depths'][data_indexes]
                        #self.data['2D_skeletons_offsets'] = self.data['2D_skeletons_offsets'][data_indexes]
                        self.data['heat_maps'] = self.data['heat_maps'][data_indexes]
                        self.data['skeletons_2D'] = self.data['skeletons_2D'][data_indexes]
                        self.data['skeletons_depths'] = self.data['skeletons_depths'][data_indexes]
                        self.data['rotation_matrix_inverted'] = self.data['rotation_matrix_inverted'][data_indexes]
                    
                point_clouds = self.data['point_clouds']
                skeletons = self.data['skeletons']
                
                batch_end = self.batch_size - len(point_clouds_batched)
                point_clouds_batched = np.append(point_clouds_batched, point_clouds[0 : batch_end], 0)
                skeletons_batched = np.append(skeletons_batched, skeletons[0 : batch_end], 0)

                if 'heat_maps' in self.data:
                    heatmaps = self.data['heat_maps']
                    skeletons_2D = self.data['skeletons_2D']
                    skeletons_depths = self.data['skeletons_depths']
                    rotation_matrix_inverted = self.data['rotation_matrix_inverted'] 
                    #skeletons_offsets = self.data['2D_skeletons_offsets']
                    #skeleton_depths = self.data['2D_skeletons_depths']
                    #skeletons_offsets_batched = np.append(skeletons_offsets_batched,
                                                          #skeletons_offsets[0 : batch_end], 0)
                    #skeleton_depths_batched = np.append(skeleton_depths_batched,
                                                        #skeleton_depths[0 : batch_end], 0)
                    heatmaps_batched = np.append(heatmaps_batched, heatmaps[0 : batch_end], 0)
                    skeletons_2D_batched = np.append(skeletons_2D_batched, skeletons_2D[0 : batch_end], 0)
                    skeletons_depths_batched = np.append(skeletons_depths_batched, skeletons_depths[0 : batch_end], 0)
                    rotation_matrix_inverted_batched = np.append(rotation_matrix_inverted_batched, rotation_matrix_inverted[0 : batch_end], 0)

                self.batch_start = batch_end

            batched_data = {key : value for key, value in self.data.items()}
            batched_data['point_clouds'] = point_clouds_batched
            batched_data['skeletons'] = skeletons_batched
            if 'heat_maps' in self.data:
                batched_data['heat_maps'] = heatmaps_batched
                #batched_data['2D_skeletons_offsets'] = skeletons_offsets_batched
                #batched_data['2D_skeletons_depths'] = skeleton_depths_batched
                batched_data['skeletons_2D'] = skeletons_2D_batched
                batched_data['skeletons_depths'] = skeletons_depths_batched
                batched_data['rotation_matrix_inverted'] =  rotation_matrix_inverted_batched
            
            if batch_end >= len(self.data['point_clouds']):
                self.batch_start = 0
                self.index += 1
                self.data = None
            else:
                self.batch_start = batch_end
            if self.transforms:
                batched_data = self.transforms(batched_data)
            if self.device != 'cpu':
                for key, value in batched_data.items():
                    if key != 'name':
                        batched_data[key] = value.to(self.device)
            return batched_data
  
    def __init__(self, dataset, batch_size, transforms, shuffle_sequences=True, shuffle_frames=True, device='cpu'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_sequences = shuffle_sequences
        self.shuffle_frames = shuffle_frames
        self.device = device
        self.transforms = transforms

    def __iter__(self):
        return self.__Iterator(self.dataset, self.batch_size, self.transforms, self.shuffle_sequences, self.shuffle_frames, self.device)"""

class SequenceDataLoader:
    class __Iterator:
        def __init__(self, dataset, batch_size, transforms=None, shuffle_sequences=True, shuffle_frames=True, device='cpu'):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle_sequences = shuffle_sequences
            self.shuffle_frames = shuffle_frames
            self.device = device
            self.transforms = transforms
            
            self.data = None
            self.index = 0
            self.batch_start = 0
            
            if shuffle_sequences:
                self.indexes = np.random.permutation(len(dataset))
            else:
                self.indexes = np.arange(len(dataset))
            
        def __next__(self):
            if self.index >= len(self.indexes):
                raise StopIteration

            if self.data is None:
                index = self.indexes[self.index]
                data = self.data = self.dataset[index]
                n = len(data['point_clouds'])
                if self.shuffle_frames:
                    frame_order = np.random.permutation(n)
                    for key, value in data.items():
                        if len(value.shape) >= 3:
                            data[key] = value[frame_order]
            data = self.data
            n = len(data['point_clouds'])
            batch_start = self.batch_start
            batch_end = batch_start + self.batch_size
            batched_data = {}
            batch_size = 0
            for key, value in data.items():
                if len(value.shape) >= 3:
                    batched_data[key] = value[batch_start : batch_end]
                    batch_size = len(batched_data[key])
                else:
                    batched_data[key] = value
            if batch_end >= n:
                self.batch_start = 0
                self.data = None
                self.index += 1
            else:
                self.batch_start = batch_end
                
            if self.transforms:
                batched_data = self.transforms(batched_data)
            if self.device != 'cpu':
                for key, value in batched_data.items():
                    if type(value) is torch.Tensor and key != 'name':
                        batched_data[key] = value.to(self.device)
            return batched_data
                    

    def __init__(self, dataset, batch_size, transforms, shuffle_sequences=True, shuffle_frames=True, device='cpu'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_sequences = shuffle_sequences
        self.shuffle_frames = shuffle_frames
        self.device = device
        self.transforms = transforms

    def __iter__(self):
        return self.__Iterator(self.dataset, self.batch_size, self.transforms, self.shuffle_sequences, self.shuffle_frames, self.device)
    
