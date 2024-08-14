import torch
import os 

from Networks import ResNet
from Networks import PoseNet
from Dataset import *
from Transforms import *
from Trainer  import *
from utils import *


CHANNELS = 3
CLASSES = 24
SEGMENTATION = False

BATCH_SIZE = 8
DATASET_PATH = os.path.join('..', 'codes', 'Dataset')

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
    min_, max_ = np.load(os.path.join(DATASET_PATH, 'min_max.npy'))
    mean = np.load(os.path.join(DATASET_PATH, 'mean.npy'))
    transform = transforms.Compose([ZeroCenter(mean),
                                    Rescale(min_, max_, -1, 1),
                                    ToTensor()])

    ACCAD_ts_dataset = Poses3D(DATASET_PATH, 
                               'ACCAD', 
                               subset='testing', 
                               transform=transform,
                               protocol=(0, 1), 
                               include_segmentation=SEGMENTATION)
    
    CMU_ts_dataset = Poses3D(DATASET_PATH,
                             'CMU',
                             subset='testing',
                             transform=transform,
                             protocol=(0, 1), 
                             include_segmentation=SEGMENTATION)
    
    EKUT_ts_dataset = Poses3D(DATASET_PATH,
                              'EKUT',
                              subset='testing',
                              transform=transform,
                              protocol=(0, 1), 
                              include_segmentation=SEGMENTATION)
    
    Eyes_Japan_ts_dataset = Poses3D(DATASET_PATH,
                                    'Eyes_Japan',
                                    subset='testing',
                                    transform=transform,
                                    protocol=(0, 1),
                                    include_segmentation=SEGMENTATION)


    model_path = os.path.join('models', 'PoseResNet18Mod', 'net_50.pt')
    model = PoseNet.PoseResNet18(CHANNELS, Eyes_Japan_ts_dataset.joints)  
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    """compute_accuracy(model, [ACCAD_ts_dataset,
                             CMU_ts_dataset,
                             EKUT_ts_dataset,
                             Eyes_Japan_ts_dataset])"""
    
    ts_dataloader = DatasetLoader([CMU_ts_dataset, 
                                   EKUT_ts_dataset, 
                                   ACCAD_ts_dataset, 
                                   Eyes_Japan_ts_dataset], 
                                   batch_size=32, 
                                   transforms=ToDevice(device),
                                   shuffle=False)

    for batched_sample in ts_dataloader:    
        sequences = batched_sample['sequences']
        targets = batched_sample['key_points']
        b, c, h, w = sequences.shape
        skeletons_targets  = batched_sample['key_points']
        predicted_joints = model(sequences)

        predictions = np.moveaxis(predicted_joints.detach().cpu().numpy(), 1, 2)
        targets = np.moveaxis(targets.detach().cpu().numpy(), 1, 2)
        sequences = np.moveaxis(sequences.detach().cpu().numpy(), 1, -1)
        
        for pt_cloud, prediction, target in zip(sequences, predictions, targets):
            tres = pt_cloud[0, 0]
            binary = pt_cloud != tres
            plt.imshow(binary[:, :, 0])
            plt.show()
            plt.imshow(binary[:, :, 1])
            plt.show()
            plt.imshow(binary[:, :, 2])
            plt.show()
            ax = plot_body(pt_cloud)
            plot_skeleton(target, ax)
            
            ax = plot_body(pt_cloud)
            plot_skeleton(prediction, ax)
            plt.show()
       