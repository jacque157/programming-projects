import torch

from Networks import PoseNet
from Dataset import *
from DatasetLoader import *
from Transforms import *
from Tester import RelativePoseTester


CHANNELS = 3
CLASSES = 24
SEGMENTATION = False
DATASET_PATH = os.path.join('..', 'codes', 'Dataset')
MINMAX_PATH = os.path.join(DATASET_PATH, 'camera_1_min_max.npy')
MEAN_PATH = os.path.join(DATASET_PATH, 'camera_1_mean.npy')
MODEL_PATH = os.path.join('.', 'models', 'PoseResNet18Mod_RelativePoseEstimation_Camera1_lr=1e-3', 'net_90.pt')


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'Testing on {device}')
    min_, max_ = np.load(MINMAX_PATH) #find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    mean = np.load(MEAN_PATH)

    transform = transforms.Compose([ZeroCenter(mean),
                                    Rescale(min_, max_, -1, 1),
                                    RootAlignedPose(),
                                    ToTensor()])

    ACCAD_ts_dataset = SingleView(DATASET_PATH, 
                                  'ACCAD', 
                                  subset='testing', 
                                  transform=transform,
                                  protocol=(0, 1), 
                                  include_segmentation=SEGMENTATION,
                                  views=(1, ))
    
    CMU_ts_dataset = SingleView(DATASET_PATH,
                                'CMU',
                                subset='testing',
                                transform=transform,
                                protocol=(0, 1), 
                                include_segmentation=SEGMENTATION,
                                views=(1, ))
    
    EKUT_ts_dataset = SingleView(DATASET_PATH,
                                 'EKUT',
                                 subset='testing',
                                 transform=transform,
                                 protocol=(0, 1), 
                                 include_segmentation=SEGMENTATION,
                                 views=(1, ))
    
    Eyes_Japan_ts_dataset = SingleView(DATASET_PATH,
                                       'Eyes_Japan',
                                        subset='testing',
                                        transform=transform,
                                        protocol=(0, 1),
                                        include_segmentation=SEGMENTATION,
                                        views=(1, ))
    
    test_dataloader = DatasetLoader([CMU_ts_dataset, 
                                   EKUT_ts_dataset, 
                                   ACCAD_ts_dataset, 
                                   Eyes_Japan_ts_dataset], 
                                   batch_size=32, 
                                   transforms=ToDevice(device),
                                   shuffle=True)
    
    model = PoseNet.PoseResNet18(CHANNELS, Eyes_Japan_ts_dataset.joints - 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda:0'))
    model.to(device)
    tester = RelativePoseTester(model, test_dataloader,
                                mean=torch.tensor(mean, device=device),
                                min_=torch.tensor(min_, device=device),
                                max_=torch.tensor(max_, device=device),
                                a=torch.tensor(-1, device=device),
                                b=torch.tensor(1, device=device))
    tester.test(False)



