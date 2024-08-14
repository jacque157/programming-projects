import torch

from Networks import PoseEstimation3DNetwork
from Networks import UNet
from Networks import ResNet
from Networks import PoseNet
from Dataset import *
from DatasetLoader import *
from Transforms import *
from Trainer  import Trainer, MultiTaskTrainer, RegressionTrainer, PoseNetTrainer, RelativePoseNetTrainer, AbsolutePoseNetTrainer


CHANNELS = 3
CLASSES = 24
SEGMENTATION = False

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training on {device}')
    min_, max_ = np.load(os.path.join('Dataset', 'min_max.npy')) #find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    mean = np.load(os.path.join('Dataset', 'mean.npy'))
    print(mean)
    print(min_)
    print(max_)
    transforms = transforms.Compose([ZeroCenter(mean),
                                     Rescale(min_, max_, -1, 1),
                                     RootAlignedPose(),
                                     #Shuffle(),
                                     ToTensor()])

    ACCAD_tr_dataset = Poses3D('Dataset', 
                               'ACCAD', 
                               subset='training', 
                               transform=transforms,
                               protocol=(0, 1), 
                               include_segmentation=SEGMENTATION)
    
    ACCAD_val_dataset = Poses3D('Dataset', 
                                'ACCAD', 
                                subset='validation', 
                                transform=transforms,
                                protocol=(0, 1), 
                                include_segmentation=SEGMENTATION)
    
    CMU_tr_dataset = Poses3D('Dataset',
                             'CMU',
                             subset='training',
                             transform=transforms,
                             protocol=(0, 1), 
                             include_segmentation=SEGMENTATION)
    
    CMU_val_dataset = Poses3D('Dataset',
                              'CMU',
                              subset='validation',
                              transform=transforms,
                              protocol=(0, 1),
                              include_segmentation=SEGMENTATION)
    
    EKUT_tr_dataset = Poses3D('Dataset',
                              'EKUT',
                              subset='training',
                              transform=transforms,
                              protocol=(0, 1), 
                              include_segmentation=SEGMENTATION)
    
    EKUT_val_dataset = Poses3D('Dataset',
                               'EKUT',
                               subset='validation',
                               transform=transforms,
                               protocol=(0, 1),
                               include_segmentation=SEGMENTATION)
    
    Eyes_Japan_tr_dataset = Poses3D('Dataset',
                                    'Eyes_Japan',
                                    subset='training',
                                    transform=transforms,
                                    protocol=(0, 1),
                                    include_segmentation=SEGMENTATION)
    
    Eyes_Japan_val_dataset = Poses3D('Dataset',
                                     'Eyes_Japan',
                                     subset='validation',
                                     transform=transforms,
                                     protocol=(0, 1),
                                     include_segmentation=SEGMENTATION)
    
    dummy_dataset = DummyPose3D('Dataset',
                                'CMU',
                                transform=transforms)


    """tr_dataloader = DatasetLoader([CMU_tr_dataset,
                                   ACCAD_tr_dataset,
                                   EKUT_tr_dataset,
                                   Eyes_Japan_tr_dataset],
                                   8, ToDevice(device))
    val_dataloader = DatasetLoader([CMU_val_dataset,
                                     ACCAD_val_dataset,
                                     EKUT_val_dataset,
                                     Eyes_Japan_val_dataset],
                                     8, ToDevice(device))"""
    dummy_dataloader = DatasetLoader([dummy_dataset],
                                     8, ToDevice(device))
    two_frames = TwoFrames('Dataset',
                           'CMU',
                           transform=transforms)
    two_frames_dataloader = DatasetLoader([two_frames],
                                     2, ToDevice(device))
    
    model = PoseNet.ResNet50RelativePose(CHANNELS, Eyes_Japan_tr_dataset.joints - 1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    trainer = RelativePoseNetTrainer(model,
                                     two_frames_dataloader,
                                     two_frames_dataloader,
                                     optimizer,
                                     scheduler,
                                     experiment_name='ResNet50RelativePose_2_frames_lr=0.001',
                                     epoch=None,
                                     mean=torch.tensor(mean, device=device),
                                     min_=torch.tensor(min_, device=device),
                                     max_=torch.tensor(max_, device=device),
                                     a=torch.tensor(-1, device=device),
                                     b=torch.tensor(1, device=device))
    trainer.train(1000)



# expriment 2 batch norm (batchnorm1d, batchnorm2d) + permutaions of frames (shuffle) + scheduler + dropout 0.25 each + LeakyReLU
# experiment 3 same as before but fixes batch of 32 instead of whole sequence (better stability?)
# experiment 4 regresion withou relative coordinates of key points, removed Local Responsenorm, fail
# experiment 5 same as 4 with Mean Per Joint Position Error as loss function

### multi_task_experiment_1 solving segmentation and pose estimation
"""model = UNet.UNetMultitask(CHANNELS, CLASSES, ACCAD_tr_dataset.joints)#PoseEstimation3DNetwork.NetworkBatchNorm(ACCAD_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    trainer = MultiTaskTrainer(model,
                               
                               [CMU_tr_dataset,
                                ACCAD_tr_dataset,
                                EKUT_tr_dataset,
                                Eyes_Japan_tr_dataset],
                               
                               [CMU_val_dataset,
                                ACCAD_val_dataset,
                                EKUT_val_dataset,
                                Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='multi_task_experiment_1',
                               batch_size=32)
    trainer.train(1000)"""


### unet regression task, pdj 0.25, half the parameters in encoder, 4 hidden layers in regression network, xavier init, batch=64
""" model = UNet.UNetPoseEstimation(CHANNELS, ACCAD_tr_dataset.joints)#PoseEstimation3DNetwork.NetworkBatchNorm(ACCAD_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    trainer = RegressionTrainer(model,
                               
                               [CMU_tr_dataset,
                                ACCAD_tr_dataset,
                                EKUT_tr_dataset,
                                Eyes_Japan_tr_dataset],
                               
                               [CMU_val_dataset,
                                ACCAD_val_dataset,
                                EKUT_val_dataset,
                                Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='U-Net_regression_experiment_1',
                               batch_size=64,
                               epoch=None)
    trainer.train(1000)"""

### multi_task_experiment_2 solving segmentation and pose estimation, pdj 0.25, half the parameters in MultiTaskUnet, xavier init, batch=64
"""model = UNet.UNetMultitaskHalf(CHANNELS, CLASSES, ACCAD_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    trainer = MultiTaskTrainer(model,
                               
                               [CMU_tr_dataset,
                                ACCAD_tr_dataset,
                                EKUT_tr_dataset,
                                Eyes_Japan_tr_dataset],
                               
                               [CMU_val_dataset,
                                ACCAD_val_dataset,
                                EKUT_val_dataset,
                                Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='multi_task_experiment_2',
                               batch_size=64,
                               epoch=None)
    trainer.train(1000)"""


### multi_task_experiment_3 solving segmentation and pose estimation, pdj 0.25, added layers in regression, leakyrelu, xavier init, batch=8
"""model = UNet.UNetMultitask2(CHANNELS, CLASSES, ACCAD_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    trainer = MultiTaskTrainer(model,
                               
                               [CMU_tr_dataset,
                                ACCAD_tr_dataset,
                                EKUT_tr_dataset,
                                Eyes_Japan_tr_dataset],
                               
                               [CMU_val_dataset,
                                ACCAD_val_dataset,
                                EKUT_val_dataset,
                                Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='multi_task_experiment_3',
                               batch_size=8,
                               epoch=None)
    trainer.train(1000)"""

### multi_task_experiment_4 fixed softmax mistake, half the params, solving segmentation and pose estimation, pdj 0.25, xavier init, batch=16
"""model = UNet.UNetMultitaskHalf(CHANNELS, CLASSES, ACCAD_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    trainer = MultiTaskTrainer(model,
                               
                               [CMU_tr_dataset,
                                ACCAD_tr_dataset,
                                EKUT_tr_dataset,
                                Eyes_Japan_tr_dataset],
                               
                               [CMU_val_dataset,
                                ACCAD_val_dataset,
                                EKUT_val_dataset,
                                Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='multi_task_experiment_4',
                               batch_size=16,
                               epoch=None)
    trainer.train(1000)"""
### multi_task_experiment_5 solving segmentation and pose estimation, pdj 0.25, added layers in regression, leakyrelu, xavier init, batch=8, fixed softmax mistake
"""model = UNet.UNetMultitask2(CHANNELS, CLASSES, ACCAD_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    trainer = MultiTaskTrainer(model,
                               
                               [CMU_tr_dataset,
                                ACCAD_tr_dataset,
                                EKUT_tr_dataset,
                                Eyes_Japan_tr_dataset],
                               
                               [CMU_val_dataset,
                                ACCAD_val_dataset,
                                EKUT_val_dataset,
                                Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='multi_task_experiment_5',
                               batch_size=8,
                               epoch=None)
    trainer.train(1000)"""

### unet regression task, pdj 0.25, Unet encoder, 4 hidden layers in regression network, xavier init, batch=16
"""model = UNet.UnetPoseEstimation2(CHANNELS, ACCAD_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    trainer = RegressionTrainer(model,
                               
                               [CMU_tr_dataset,
                                ACCAD_tr_dataset,
                                EKUT_tr_dataset,
                                Eyes_Japan_tr_dataset],
                               
                               [CMU_val_dataset,
                                ACCAD_val_dataset,
                                EKUT_val_dataset,
                                Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='U-Net_regression_experiment_2',
                               batch_size=16,
                               epoch=None)
    trainer.train(1000)"""

### unet regression task dummy dataset, pdj 0.25, Unet encoder, 4 hidden layers in regression network, xavier init, batch=16
"""
    Eyes_Japan_tr_dataset = DummyPose3D('Dataset',
                                    'Eyes_Japan',
                                    subset='training',
                                    transform=transforms)
    
    Eyes_Japan_val_dataset = DummyPose3D('Dataset',
                                     'Eyes_Japan',
                                     subset='validation',
                                     transform=transforms)

    model = UNet.UnetPoseEstimation2(CHANNELS, Eyes_Japan_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    
    trainer = RegressionTrainer(model,
                               
                               [Eyes_Japan_tr_dataset],
                               
                               [Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='U-Net_regression_experiment_3',
                               batch_size=16,
                               epoch=None)
    trainer.train(1000)"""

# same as before, learning rate 0.001, scheduler 0.1
"""Eyes_Japan_tr_dataset = DummyPose3D('Dataset',
                                    'Eyes_Japan',
                                    subset='training',
                                    transform=transforms)
    
    Eyes_Japan_val_dataset = DummyPose3D('Dataset',
                                     'Eyes_Japan',
                                     subset='validation',
                                     transform=transforms)

    model = UNet.UnetPoseEstimation2(CHANNELS, Eyes_Japan_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainer = RegressionTrainer(model,
                               
                               [Eyes_Japan_tr_dataset],
                               
                               [Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='U-Net_regression_experiment_4',
                               batch_size=16,
                               epoch=None)
    trainer.train(1000)"""

# same as before, learning rate 0.1, scheduler 0.1
"""Eyes_Japan_tr_dataset = DummyPose3D('Dataset',
                                    'Eyes_Japan',
                                    subset='training',
                                    transform=transforms)
    
    Eyes_Japan_val_dataset = DummyPose3D('Dataset',
                                     'Eyes_Japan',
                                     subset='validation',
                                     transform=transforms)

    model = UNet.UnetPoseEstimation2(CHANNELS, Eyes_Japan_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainer = RegressionTrainer(model,
                               
                               [Eyes_Japan_tr_dataset],
                               
                               [Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='U-Net_regression_experiment_5',
                               batch_size=16,
                               epoch=None)
    trainer.train(1000)"""

### Same as before, dummy dataset reduced
"""Eyes_Japan_tr_dataset = DummyPose3D('Dataset',
                                    'Eyes_Japan',
                                    subset='training',
                                    transform=transforms)
    
    Eyes_Japan_val_dataset = DummyPose3D('Dataset',
                                     'Eyes_Japan',
                                     subset='validation',
                                     transform=transforms)

    model = UNet.UnetPoseEstimation2(CHANNELS, Eyes_Japan_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainer = RegressionTrainer(model,
                               
                               [Eyes_Japan_tr_dataset],
                               
                               [Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='U-Net_regression_experiment_6',
                               batch_size=16,
                               epoch=None)
    trainer.train(1000)"""

### Same as before, no Random Crop
"""Eyes_Japan_tr_dataset = DummyPose3D('Dataset',
                                    'Eyes_Japan',
                                    subset='training',
                                    transform=transforms)
    
    Eyes_Japan_val_dataset = DummyPose3D('Dataset',
                                     'Eyes_Japan',
                                     subset='validation',
                                     transform=transforms)

    model = UNet.UnetPoseEstimation2(CHANNELS, Eyes_Japan_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainer = RegressionTrainer(model,
                               
                               [Eyes_Japan_tr_dataset],
                               
                               [Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='U-Net_regression_experiment_6',
                               batch_size=16,
                               epoch=None)
    trainer.train(1000)"""

### !!!Changed dataset distribution!!!

### ResNet regresion, absolute positions 

"""
    transforms = transforms.Compose([ZeroCenter(),
                                     Rescale(min_, max_, -1, 1),
                                     
                                     Shuffle(),
                                     ToTensor(),
                                     ToDevice(device)])

    ACCAD_tr_dataset = Poses3D('Dataset', 
                               'ACCAD', 
                               subset='training', 
                               transform=transforms,
                               protocol=(0, 1), 
                               include_segmentation=SEGMENTATION)
    
    ACCAD_val_dataset = Poses3D('Dataset', 
                                'ACCAD', 
                                subset='validation', 
                                transform=transforms,
                                protocol=(0, 1), 
                                include_segmentation=SEGMENTATION)
    
    CMU_tr_dataset = Poses3D('Dataset',
                             'CMU',
                             subset='training',
                             transform=transforms,
                             protocol=(0, 1), 
                             include_segmentation=SEGMENTATION)
    
    CMU_val_dataset = Poses3D('Dataset',
                              'CMU',
                              subset='validation',
                              transform=transforms,
                              protocol=(0, 1),
                              include_segmentation=SEGMENTATION)
    
    EKUT_tr_dataset = Poses3D('Dataset',
                              'EKUT',
                              subset='training',
                              transform=transforms,
                              protocol=(0, 1), 
                              include_segmentation=SEGMENTATION)
    
    EKUT_val_dataset = Poses3D('Dataset',
                               'EKUT',
                               subset='validation',
                               transform=transforms,
                               protocol=(0, 1),
                               include_segmentation=SEGMENTATION)
    
    Eyes_Japan_tr_dataset = Poses3D('Dataset',
                                    'Eyes_Japan',
                                    subset='training',
                                    transform=transforms,
                                    protocol=(0, 1),
                                    include_segmentation=SEGMENTATION)
    
    Eyes_Japan_val_dataset = Poses3D('Dataset',
                                     'Eyes_Japan',
                                     subset='validation',
                                     transform=transforms,
                                     protocol=(0, 1),
                                     include_segmentation=SEGMENTATION)
    
    model = ResNet.ResNet(CHANNELS, Eyes_Japan_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    trainer = RegressionTrainer(model,
                               
                               [CMU_tr_dataset,
                                ACCAD_tr_dataset,
                                EKUT_tr_dataset,
                                Eyes_Japan_tr_dataset],
                               
                               [CMU_val_dataset,
                                ACCAD_val_dataset,
                                EKUT_val_dataset,
                                Eyes_Japan_val_dataset],
                               
                               optimizer,
                               scheduler,
                               experiment_name='ResNet_1',
                               batch_size=16,
                               epoch=None)
    trainer.train(1000)
"""


### Resnet direct joint estimation, validation metric is not calculated properly
"""transforms = transforms.Compose([ZeroCenter(),
                                     Rescale(min_, max_, -1, 1),
                                     
                                     Shuffle(),
                                     ToTensor(),
                                     ToDevice(device)])

    ...

    model = ResNet.ResNet(CHANNELS, Eyes_Japan_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    trainer = RegressionTrainer(model,
                                
                                [CMU_tr_dataset,
                                ACCAD_tr_dataset,
                                EKUT_tr_dataset,
                                Eyes_Japan_tr_dataset],
                                
                                [CMU_val_dataset,
                                ACCAD_val_dataset,
                                EKUT_val_dataset,
                                Eyes_Japan_val_dataset],
                                
                                optimizer,
                                scheduler,
                                experiment_name='ResNet_1',
                                batch_size=16,
                                epoch=26)
    trainer.train(1000)
"""


### 3D Heatmap generation with "PoseNet18", new datasetloader, data is zero centered absolute position cannot be infered
"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training on {device}')
    min_, max_ = find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    mean = np.load(os.path.join('Dataset', 'mean.npy'))
    transforms = transforms.Compose([ZeroCenter(mean),
                                     Rescale(min_, max_, -1, 1),
                                     RootAlignedPose(),
                                     Shuffle(),
                                     ToTensor()])

    ACCAD_tr_dataset = Poses3D('Dataset', 
                               'ACCAD', 
                               subset='training', 
                               transform=transforms,
                               protocol=(0, 1), 
                               include_segmentation=SEGMENTATION)
    
    ACCAD_val_dataset = Poses3D('Dataset', 
                                'ACCAD', 
                                subset='validation', 
                                transform=transforms,
                                protocol=(0, 1), 
                                include_segmentation=SEGMENTATION)
    
    CMU_tr_dataset = Poses3D('Dataset',
                             'CMU',
                             subset='training',
                             transform=transforms,
                             protocol=(0, 1), 
                             include_segmentation=SEGMENTATION)
    
    CMU_val_dataset = Poses3D('Dataset',
                              'CMU',
                              subset='validation',
                              transform=transforms,
                              protocol=(0, 1),
                              include_segmentation=SEGMENTATION)
    
    EKUT_tr_dataset = Poses3D('Dataset',
                              'EKUT',
                              subset='training',
                              transform=transforms,
                              protocol=(0, 1), 
                              include_segmentation=SEGMENTATION)
    
    EKUT_val_dataset = Poses3D('Dataset',
                               'EKUT',
                               subset='validation',
                               transform=transforms,
                               protocol=(0, 1),
                               include_segmentation=SEGMENTATION)
    
    Eyes_Japan_tr_dataset = Poses3D('Dataset',
                                    'Eyes_Japan',
                                    subset='training',
                                    transform=transforms,
                                    protocol=(0, 1),
                                    include_segmentation=SEGMENTATION)
    
    Eyes_Japan_val_dataset = Poses3D('Dataset',
                                     'Eyes_Japan',
                                     subset='validation',
                                     transform=transforms,
                                     protocol=(0, 1),
                                     include_segmentation=SEGMENTATION)
    
    tr_dataloader = DatasetLoader([CMU_tr_dataset,
                                   ACCAD_tr_dataset,
                                   EKUT_tr_dataset,
                                   Eyes_Japan_tr_dataset],
                                   16, ToDevice(device))
    val_dataloader = DatasetLoader([CMU_val_dataset,
                                     ACCAD_val_dataset,
                                     EKUT_val_dataset,
                                     Eyes_Japan_val_dataset],
                                     16, ToDevice(device))
    
    model = PoseNet.PoseNet18(CHANNELS, Eyes_Japan_tr_dataset.joints)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    trainer = PoseNetTrainer(model,
                             tr_dataloader,
                             val_dataloader,
                             optimizer,
                             scheduler,
                             experiment_name='PoseNet-18',
                             epoch=None)
    trainer.train(1000)
"""

# relative Pose estimation with RelativePoseNet50/18 
"""    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training on {device}')
    min_, max_ = find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    mean = np.load(os.path.join('Dataset', 'mean.npy'))
    transforms = transforms.Compose([ZeroCenter(mean),
                                     Rescale(min_, max_, -1, 1),
                                     RootAlignedPose(),
                                     Shuffle(),
                                     ToTensor()])

    ACCAD_tr_dataset = Poses3D('Dataset', 
                               'ACCAD', 
                               subset='training', 
                               transform=transforms,
                               protocol=(0, 1), 
                               include_segmentation=SEGMENTATION)
    
    ACCAD_val_dataset = Poses3D('Dataset', 
                                'ACCAD', 
                                subset='validation', 
                                transform=transforms,
                                protocol=(0, 1), 
                                include_segmentation=SEGMENTATION)
    
    CMU_tr_dataset = Poses3D('Dataset',
                             'CMU',
                             subset='training',
                             transform=transforms,
                             protocol=(0, 1), 
                             include_segmentation=SEGMENTATION)
    
    CMU_val_dataset = Poses3D('Dataset',
                              'CMU',
                              subset='validation',
                              transform=transforms,
                              protocol=(0, 1),
                              include_segmentation=SEGMENTATION)
    
    EKUT_tr_dataset = Poses3D('Dataset',
                              'EKUT',
                              subset='training',
                              transform=transforms,
                              protocol=(0, 1), 
                              include_segmentation=SEGMENTATION)
    
    EKUT_val_dataset = Poses3D('Dataset',
                               'EKUT',
                               subset='validation',
                               transform=transforms,
                               protocol=(0, 1),
                               include_segmentation=SEGMENTATION)
    
    Eyes_Japan_tr_dataset = Poses3D('Dataset',
                                    'Eyes_Japan',
                                    subset='training',
                                    transform=transforms,
                                    protocol=(0, 1),
                                    include_segmentation=SEGMENTATION)
    
    Eyes_Japan_val_dataset = Poses3D('Dataset',
                                     'Eyes_Japan',
                                     subset='validation',
                                     transform=transforms,
                                     protocol=(0, 1),
                                     include_segmentation=SEGMENTATION)
    
    tr_dataloader = DatasetLoader([CMU_tr_dataset,
                                   ACCAD_tr_dataset,
                                   EKUT_tr_dataset,
                                   Eyes_Japan_tr_dataset],
                                   8, ToDevice(device))
    val_dataloader = DatasetLoader([CMU_val_dataset,
                                     ACCAD_val_dataset,
                                     EKUT_val_dataset,
                                     Eyes_Japan_val_dataset],
                                     8, ToDevice(device))
    
    model = PoseNet.RelativePoseNet50(CHANNELS, Eyes_Japan_tr_dataset.joints - 1)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    trainer = RelativePoseNetTrainer(model,
                                     tr_dataloader,
                                     val_dataloader,
                                     optimizer,
                                     scheduler,
                                     experiment_name='PoseNet-18-Relative',
                                     epoch=None)
    trainer.train(1000)"""