import torch

from Networks import PoseNet
from Dataset import *
from DatasetLoader import *
from Transforms import *
from Trainer  import RelativePoseNetTrainer, AbsolutePoseNetTrainer


CHANNELS = 3
CLASSES = 24
SEGMENTATION = False
DATASET_PATH = os.path.join('..', 'codes', 'Dataset')
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training on {device}')
    min_, max_ = np.load(os.path.join(DATASET_PATH, 'min_max.npy')) #find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    mean = np.load(os.path.join(DATASET_PATH, 'mean.npy'))

    transform = transforms.Compose([#ZeroCenter(mean),
                                    #Rescale(min_, max_, -1, 1),
                                    RootAlignedPose(),
                                    ToTensor()])

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
    
    """dummy_dataset = DummyPose3D(DATASET_PATH,
                                'CMU',
                                transform=transform)

    dummy_dataloader = DatasetLoader([dummy_dataset],
                                     8, ToDevice(device))
    
    
    samples = TwoFrames(DATASET_PATH, 'CMU', transform=None)[0]['sequences']

    mean = np.mean(samples, (0, 1, 2))
    min_ = np.min(samples - mean, (0, 1, 2))
    max_ = np.max(samples - mean, (0, 1, 2))
    
    print(mean.shape)

    transform = transforms.Compose([#RootAlignedPose(),
                                     ZeroCenter(mean),
                                     Rescale(min_, max_, -1, 1),
                                     #Shuffle(),
                                     ToTensor()])
    two_frames = TwoFrames(DATASET_PATH,
                           'CMU',
                           transform=transform)
    
    two_frames_dataloader = DatasetLoader([two_frames],
                                     2, ToDevice(device),
                                     shuffle=False)"""
    
    tr_dataloader = DatasetLoader([CMU_tr_dataset, 
                                   EKUT_tr_dataset, 
                                   ACCAD_tr_dataset, 
                                   Eyes_Japan_tr_dataset], 
                                   batch_size=32, 
                                   transforms=ToDevice(device),
                                   shuffle=True)
    
    val_dataloader = DatasetLoader([CMU_val_dataset, 
                                   EKUT_val_dataset, 
                                   ACCAD_val_dataset, 
                                   Eyes_Japan_val_dataset], 
                                   batch_size=32, 
                                   transforms=ToDevice(device),
                                   shuffle=True)

    model = PoseNet.PoseResNet18(CHANNELS, Eyes_Japan_tr_dataset.joints - 1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    trainer = RelativePoseNetTrainer(model,
                                     tr_dataloader,
                                     val_dataloader,
                                     optimizer,
                                     scheduler,
                                     experiment_name='PoseResNet18ModRelativePoseEstimationNoPreprocessinglr=0.0001',
                                     epoch=12,
                                     save_states=True,
                                     mean=torch.tensor(mean, device=device),
                                     min_=torch.tensor(min_, device=device),
                                     max_=torch.tensor(max_, device=device),
                                     a=torch.tensor(-1, device=device),
                                     b=torch.tensor(1, device=device))
    trainer.train(1000)



