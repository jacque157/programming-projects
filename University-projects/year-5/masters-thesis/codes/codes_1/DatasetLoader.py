from Dataset import *
from Transforms import * 


class DatasetLoader:
    class __Iterator:
        def __init__(self, datasets, batch_size, transforms=None):
            self.datasets = datasets
            self.num_of_datasets = len(datasets)
            self.batch_size = batch_size
            self.transforms = transforms
            self.include_segmentation = datasets[0].include_segmentation

            self.dataset_index = 0
            self.seq_number = 0
            self.frame_index = 0

            self.sample = self.datasets[self.dataset_index][self.seq_number]
            self.frames = len(self.sample['sequences'])
            self.name = self.datasets[self.dataset_index].name

        def __next__(self):
            if self.frame_index >= self.frames:
                self.frame_index = 0
                self.seq_number += 1
                current_dataset = self.datasets[self.dataset_index]
                if self.seq_number >= len(current_dataset):
                    self.dataset_index += 1
                    self.seq_number = 0
                    if self.dataset_index >= self.num_of_datasets:
                        raise StopIteration()
                    else:
                        current_dataset = self.datasets[self.dataset_index]
                        self.name = self.datasets[self.dataset_index].name
                self.sample = current_dataset[self.seq_number]
                self.frames = len(self.sample['sequences'])


            start = self.frame_index
            end = start + self.batch_size
            
            batched_poses = self.sample['sequences'][start : end]
            batched_masks = self.sample['valid_points'][start : end]
            batched_skeletons = self.sample['key_points'][start : end]
            batched_centres = self.sample['root_key_points'][start : end]


            batched_sample = {'sequences' : batched_poses, 
                              'valid_points' : batched_masks, 
                              'key_points' : batched_skeletons, 
                              'root_key_points' : batched_centres,
                              'name' : self.name}
            
            if self.include_segmentation:
                batched_segmentation = self.sample['segmentations'][start : end]
                batched_sample['segmentations'] = batched_segmentation

            self.frame_index += self.batch_size

            if self.transforms:
                return self.transforms(batched_sample)
            return batched_sample

    def __init__(self, datasets, batch_size, transforms=None):
        self.datasets = datasets
        self.num_of_datasets = len(datasets)
        self.batch_size = batch_size
        self.transforms = transforms

    def __iter__(self):
        return self.__Iterator(self.datasets, self.batch_size, self.transforms)
    

"""sample = {'sequences' : poses, 'valid_points' : mask, 'key_points' : skeletons, 'root_key_points' : centres}
        if self.include_segmentation:
            sample['segmentations'] = segmentations"""



if __name__ == '__main__':
    SEGMENTATION = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training on {device}')
    min_, max_ = find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    mean = np.load(os.path.join('Dataset', 'mean.npy'))
    transforms = transforms.Compose([ZeroCenter(mean),
                                     Rescale(min_, max_, -1, 1),
                                     RootAlignedPose(),
                                     Shuffle(),
                                     ToTensor(),
                                     ])

    Eyes_tr_dataset = Poses3D('Dataset', 
                               'Eyes_Japan', 
                               subset='tr', 
                               transform=transforms,
                               protocol=(0, 1), 
                               include_segmentation=SEGMENTATION)
    
    EKUT_tr_dataset = Poses3D('Dataset',
                              'EKUT',
                              subset='tr',
                              transform=transforms,
                              protocol=(0, 1), 
                              include_segmentation=SEGMENTATION)
    
    data = DatasetLoader([Eyes_tr_dataset, EKUT_tr_dataset], 32, ToDevice(device))

    #for batch in data:
        #print(batch['name'])
    print(len(data))