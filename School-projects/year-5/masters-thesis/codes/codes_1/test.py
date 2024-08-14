import torch
import os 

from Networks import PoseEstimation3DNetwork
from Networks import UNet
from Networks import ResNet
from Dataset import *
from Transforms import *
from Trainer  import *
from utils import *


CHANNELS = 3
CLASSES = 24
SEGMENTATION = False

BATCH_SIZE = 8

def distance(predictions, targets):
    return torch.sqrt(torch.sum((predictions - targets) ** 2, 1))
     
def compute_accuracy(model, datasets):
    n = 0
    sum_ = 0
    with torch.no_grad():
        for dataset in datasets:
            dataset_n = 0
            dataset_sum = 0
            for sample in dataset:
                sequences = sample['sequences']
                targets = sample['key_points']
                
                for i in range(0, len(sequences), BATCH_SIZE):
                    sequences_batch = sequences[i:i+BATCH_SIZE]
                    targets_batch = targets[i:i+BATCH_SIZE]
                    predictions = model(sequences_batch)
                    b, j, c = targets_batch.shape
                
                    dataset_n += b * j
                    dataset_sum += torch.sum(distance(predictions, targets_batch))
            n += dataset_n
            sum_ += dataset_sum
            print(f"Datset: {dataset.name} Accuracy: {dataset_sum / dataset_n}")
        print(f"Accuracy: {sum_ / n}")
    return sum_ / n
            
            
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    min_, max_ = find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    transforms = transforms.Compose([ZeroCenter(),
                                     Rescale(min_, max_, -1, 1),
                                     #RandomCrop((224, 224), (257, 344)),
                                     #Crop((224, 224)),
                                     #RelativeJointsPosition(),
                                     #ZeroOutEntries(),
                                     ToTensor(),
                                     ToDevice(device),
                                     AutoGrad(False)
                                     ])

    ACCAD_ts_dataset = Poses3D('Dataset', 
                               'ACCAD', 
                               subset='testing', 
                               transform=transforms,
                               protocol=(0, 1), 
                               include_segmentation=SEGMENTATION)
    
    CMU_ts_dataset = Poses3D('Dataset',
                             'CMU',
                             subset='testing',
                             transform=transforms,
                             protocol=(0, 1), 
                             include_segmentation=SEGMENTATION)
    
    EKUT_ts_dataset = Poses3D('Dataset',
                              'EKUT',
                              subset='testing',
                              transform=transforms,
                              protocol=(0, 1), 
                              include_segmentation=SEGMENTATION)
    
    Eyes_Japan_ts_dataset = Poses3D('Dataset',
                                    'Eyes_Japan',
                                    subset='testing',
                                    transform=transforms,
                                    protocol=(0, 1),
                                    include_segmentation=SEGMENTATION)

    ACCAD_ts_dataset_raw = Poses3D('Dataset', 
                                   'ACCAD', 
                                   subset='testing', 
                                   transform=None,
                                   protocol=(0, 1), 
                                   include_segmentation=SEGMENTATION)
    
    CMU_ts_dataset_raw = Poses3D('Dataset',
                                 'CMU',
                                 subset='testing',
                                 transform=None,
                                 protocol=(0, 1), 
                                 include_segmentation=SEGMENTATION)
    
    EKUT_ts_dataset_raw = Poses3D('Dataset',
                                  'EKUT',
                                  subset='testing',
                                  transform=None,
                                  protocol=(0, 1),
                                  include_segmentation=SEGMENTATION)
    
    Eyes_Japan_ts_dataset_raw = Poses3D('Dataset',
                                        'Eyes_Japan',
                                        subset='testing',
                                        transform=None,
                                        protocol=(0, 1),
                                        include_segmentation=SEGMENTATION)
    
    Eyes_Japan_tr_dataset = DummyPose3D('Dataset',
                                        'Eyes_Japan',
                                        subset='training',
                                        transform=transforms)
    
    
    """model_path = os.path.join('models', 'multi_task_experiment_5', 'net_14.pt')
    model = UNet.UNetMultitask2(CHANNELS, CLASSES, ACCAD_ts_dataset.joints)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    sample_idx = 7
    sample = CMU_ts_dataset[sample_idx] #Eyes_Japan_ts_dataset[idx]
    sequences = torch.squeeze(sample['sequences'], 0)
    predictions, segmentation = model(sequences)
    predictions = predictions.detach().cpu()
    segmentation = segmentation.detach().cpu()
    """
    model_path = os.path.join('models', 'ResNet_1', 'net_80.pt')
    model = ResNet.ResNet(CHANNELS, Eyes_Japan_tr_dataset.joints)    
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    """compute_accuracy(model, [ACCAD_ts_dataset,
                             CMU_ts_dataset,
                             EKUT_ts_dataset,
                             Eyes_Japan_ts_dataset])"""
    
    sample_idx = 7
    sample = CMU_ts_dataset[sample_idx] #Eyes_Japan_ts_dataset[idx]
    sample_raw = CMU_ts_dataset_raw[sample_idx]
    sequences = torch.squeeze(sample['sequences'], 0)
    sequences_raw = sample_raw['sequences']
    #predictions, segmentation = model(sequences)
    predictions = model(sequences)
    predictions = predictions.detach().cpu()
    
    
    idx = 10
    predictions = np.moveaxis(predictions.numpy(), 1, 2)
    sequences = sample_raw['sequences']
    #sequences = np.moveaxis(sequences, 1, 3)
    targets = sample['key_points'].detach().cpu()
    targets = np.moveaxis(targets.numpy(), 1, 2)
    centres = sample['root_key_points'].detach().cpu()

    for idx in range(len(targets)):
        target_skeleton = targets[idx]#reconstruct_skeleton(targets[idx], centres[idx])
        predicted_skeleton = predictions[idx]# reconstruct_skeleton(predictions[idx], centres[idx])
        ax = plot_body(sequences[idx])
        #plot_skeleton(targets[idx], ax)
        plot_skeleton(target_skeleton, ax)
        
        ax = plot_body(sequences[idx])
        plot_skeleton(predicted_skeleton, ax)
        #plot_skeleton(predictions[idx], ax)
        plt.show()
    

    
