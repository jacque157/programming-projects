import torch
from Dataset import *


CHANNELS = 3
CLASSES = 24
SEGMENTATION = False
DATASET_PATH = os.path.join('..', 'codes', 'Dataset')


def get_size(dataset):
    frames = 0
    for sample in dataset:
        sequences = sample['sequences']
        s, h, w, c = sequences.shape
        frames += s

    print(f"Dataset: {dataset.name}, sequences: {len(dataset)}, frames: {frames}")

if __name__ == '__main__':
    ACCAD_dataset = SingleView(DATASET_PATH, 
                                  'ACCAD', 
                                  subset='all', 
                                  transform=None,
                                  protocol=(0, 1), 
                                  include_segmentation=SEGMENTATION,
                                  views=(1, ))
    
    CMU_dataset = SingleView(DATASET_PATH,
                                'CMU',
                                subset='all',
                                transform=None,
                                protocol=(0, 1), 
                                include_segmentation=SEGMENTATION,
                                views=(1, ))
    
    EKUT_dataset = SingleView(DATASET_PATH,
                                 'EKUT',
                                 subset='all',
                                 transform=None,
                                 protocol=(0, 1), 
                                 include_segmentation=SEGMENTATION,
                                 views=(1, ))
    
    Eyes_Japan_dataset = SingleView(DATASET_PATH,
                                       'Eyes_Japan',
                                        subset='all',
                                        transform=None,
                                        protocol=(0, 1),
                                        include_segmentation=SEGMENTATION,
                                        views=(1, ))
    

    get_size(ACCAD_dataset)
    get_size(CMU_dataset)
    get_size(EKUT_dataset)
    get_size(Eyes_Japan_dataset)



