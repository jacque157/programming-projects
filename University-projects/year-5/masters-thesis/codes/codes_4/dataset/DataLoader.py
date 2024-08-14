import numpy as np
import torch

class SequenceDataLoader:
    class _Iterator:
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
                n = len(data['skeletons'])
                if self.shuffle_frames:
                    frame_order = np.random.permutation(n)
                    for key, value in data.items():
                        if len(value.shape) >= 3 and len(value) == n:
                            data[key] = value[frame_order]
            data = self.data
            n = len(data['skeletons'])
            batch_start = self.batch_start
            batch_end = batch_start + self.batch_size
            batched_data = {}
            batch_size = 0
            for key, value in data.items():
                if len(value.shape) >= 3 and len(value) == n:
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
        return self._Iterator(self.dataset, self.batch_size, self.transforms, self.shuffle_sequences, self.shuffle_frames, self.device)

class SingleBatchSequenceDataLoader(SequenceDataLoader):
    class _Iterator:
        def __init__(self, dataset, batch_size, index=0, transforms=None, shuffle_frames=False, device='cpu'):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle_frames = shuffle_frames
            self.device = device
            self.transforms = transforms

            self.start = index  
            self.index = index
                
        def __next__(self):
            if self.index != self.start :
                raise StopIteration
            else:
                self.index += 1
            data = self.dataset[self.index]
            n = len(data['skeletons'])
            if self.shuffle_frames:
                frame_order = np.random.permutation(n)
                for key, value in data.items():
                    if len(value.shape) >= 3:
                        data[key] = value[frame_order]

            batched_data = {}                              
            for key, value in data.items():
                if len(value.shape) >= 3:
                    batched_data[key] = value[0 : self.batch_size]
                else:
                    batched_data[key] = value
                
            if self.transforms:
                batched_data = self.transforms(batched_data)
            if self.device != 'cpu':
                for key, value in batched_data.items():
                    if type(value) is torch.Tensor and key != 'name':
                        batched_data[key] = value.to(self.device)
            return batched_data

    def __init__(self, dataset, batch_size, index, transforms=None, shuffle_frames=False, device='cpu'):
        super().__init__(dataset, batch_size, transforms, shuffle_sequences=False, shuffle_frames=shuffle_frames, device=device)
        self.index = index

    def __iter__(self):
        return self._Iterator(self.dataset, self.batch_size, self.index, self.transforms, self.shuffle_frames, self.device)
        

class PartitionSequenceDataLoader(SequenceDataLoader):
    class _Iterator(SequenceDataLoader._Iterator):
        def __init__(self, dataset, batch_size, maximum=-1, transforms=None, shuffle_sequences=True, shuffle_frames=True, device='cpu'):
            super().__init__(dataset, batch_size, transforms=transforms, shuffle_sequences=shuffle_sequences, shuffle_frames=shuffle_frames, device=device)
            self.maximum = maximum
            self.i = 0

        def __next__(self):
            if self.i >= self.maximum:
                raise StopIteration
            self.i += 1
            return super().__next__()
        
    def __init__(self, dataset, batch_size, maximum, transforms=None, shuffle_sequences=True, shuffle_frames=True, device='cpu'):
        super().__init__(dataset, batch_size, transforms=transforms, shuffle_sequences=shuffle_sequences, shuffle_frames=shuffle_frames, device=device)
        self.maximum = maximum

    def __iter__(self):
        return self._Iterator(self.dataset, self.batch_size, self.maximum, self.transforms, self.shuffle_sequences, self.shuffle_frames, self.device)
        
