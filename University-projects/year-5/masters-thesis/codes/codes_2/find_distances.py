import torch

from Dataset import *
from Transforms import *

SEGMENTATION = False



# Min joints distance = 59.68647766113281
# Max joints distance = 1207.5904541015625

if __name__ == '__main__':
    device =  "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training on {device}')
    transforms = transforms.Compose([ToTensor(),
                                     ToDevice(device)])

    ACCAD_dataset = Poses3D('Dataset', 
                               'ACCAD', 
                               subset=None, 
                               transform=transforms,
                               protocol=(0, 1), 
                               include_segmentation=SEGMENTATION)
    
    CMU_dataset = Poses3D('Dataset',
                             'CMU',
                             subset=None,
                             transform=transforms,
                             protocol=(0, 1), 
                             include_segmentation=SEGMENTATION)
    
    EKUT_dataset = Poses3D('Dataset',
                              'EKUT',
                              subset=None,
                              transform=transforms,
                              protocol=(0, 1), 
                              include_segmentation=SEGMENTATION)
    
    Eyes_Japan_dataset = Poses3D('Dataset',
                                    'Eyes_Japan',
                                    subset=None,
                                    transform=transforms,
                                    protocol=(0, 1),
                                    include_segmentation=SEGMENTATION)
    

    best_min_distance = float('inf')
    best_max_distance = float('-inf')
    for dataset in [Eyes_Japan_dataset, CMU_dataset, EKUT_dataset, ACCAD_dataset]:
        print(f'dataset: {dataset.name}')

        for i, data in enumerate(dataset):
            skeletons = data['key_points']
            s, c, j = skeletons.shape
            centres = skeletons[:, :, 0]

            distances = torch.sqrt(torch.sum((skeletons - centres.view(s, c, 1)) ** 2, 1))

            min_distance = torch.min(distances[:, 1:])
            max_distance = torch.max(distances[:, 1:])
            
            #print(distances.shape)
            if min_distance < best_min_distance:
                best_min_distance = min_distance
            if max_distance > best_max_distance:
                best_max_distance = max_distance
            if i % 20 == 19:
                print(f'Currecntly min joints distance = {min_distance}, best = {best_min_distance}')
                print(f'Currecntly max joints distance = {max_distance}, best = {best_max_distance}')


    print(f'Min joints distance = {best_min_distance}') # Min joints distance = 59.68647766113281
    print(f'Max joints distance = {best_max_distance}') # Max joints distance = 1207.5904541015625
