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
MINMAX_PATH = os.path.join(DATASET_PATH, 'camera_1_min_max.npy')
MEAN_PATH = os.path.join(DATASET_PATH, 'camera_1_mean.npy')

if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'Training on {device}')
    min_, max_ = np.load(MINMAX_PATH) #find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    mean = np.load(MEAN_PATH)

    transform = transforms.Compose([ZeroCenter(mean),
                                    Rescale(min_, max_, -1, 1),
                                    RootAlignedPose(),
                                    ToTensor()])

    ACCAD_tr_dataset = SingleView(DATASET_PATH, 
                                  'ACCAD', 
                                  subset='training', 
                                  transform=transform,
                                  protocol=(0, 1), 
                                  include_segmentation=SEGMENTATION,
                                  views=(1, ))
    
    ACCAD_val_dataset = SingleView(DATASET_PATH, 
                                   'ACCAD', 
                                    subset='validation', 
                                    transform=transform,
                                    protocol=(0, 1), 
                                    include_segmentation=SEGMENTATION,
                                  views=(1, ))
    
    CMU_tr_dataset = SingleView(DATASET_PATH,
                                'CMU',
                                subset='training',
                                transform=transform,
                                protocol=(0, 1), 
                                include_segmentation=SEGMENTATION,
                                views=(1, ))
    
    CMU_val_dataset = SingleView(DATASET_PATH,
                                 'CMU',
                                 subset='validation',
                                 transform=transform,
                                 protocol=(0, 1),
                                 include_segmentation=SEGMENTATION,
                                 views=(1, ))
    
    EKUT_tr_dataset = SingleView(DATASET_PATH,
                                 'EKUT',
                                 subset='training',
                                 transform=transform,
                                 protocol=(0, 1), 
                                 include_segmentation=SEGMENTATION,
                                 views=(1, ))
    
    EKUT_val_dataset = SingleView(DATASET_PATH,
                                  'EKUT',
                                  subset='validation',
                                  transform=transform,
                                  protocol=(0, 1),
                                  include_segmentation=SEGMENTATION,
                                  views=(1, ))
    
    Eyes_Japan_tr_dataset = SingleView(DATASET_PATH,
                                       'Eyes_Japan',
                                        subset='training',
                                        transform=transform,
                                        protocol=(0, 1),
                                        include_segmentation=SEGMENTATION,
                                        views=(1, ))
    
    Eyes_Japan_val_dataset = SingleView(DATASET_PATH,
                                        'Eyes_Japan',
                                        subset='validation',
                                        transform=transform,
                                        protocol=(0, 1),
                                        include_segmentation=SEGMENTATION,
                                        views=(1, ))
    

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
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    
    trainer = RelativePoseNetTrainer(model,
                                     tr_dataloader,
                                     val_dataloader,
                                     optimizer,
                                     scheduler,
                                     experiment_name='PoseResNet18Mod_RelativePoseEstimation_Camera1',
                                     epoch=None,
                                     save_states=True,
                                     mean=torch.tensor(mean, device=device),
                                     min_=torch.tensor(min_, device=device),
                                     max_=torch.tensor(max_, device=device),
                                     a=torch.tensor(-1, device=device),
                                     b=torch.tensor(1, device=device))
    trainer.train(100)



