import os
import torch

from dataset.Dataset import MultiviewStructuredCameraPointCloudDataset as MultiviewDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Networks.AdaFuse import PoseResNet, AdaFuse

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
POSE_NET_PATH = os.path.join('models', 'PoseResNet50', 'net_20.pt')
EXPERIMENT_NAME = 'CenterNetDLA34PoseEstimationImages'
BATCH_SIZE = 16 // 16
EPOCHS = 100
LEARNING_RATE = 1e-3
SCHEDULE_STEP = 3
SCHEDULE_RATE = 0.5
REDUCTION = 4

if __name__ == '__main__':
    train_dataset = MultiviewDataset(DATASET_PATH, 'train')
    min_ = train_dataset.min(centered=True)
    max_ = train_dataset.max(centered=True)
    avg = train_dataset.avg()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=False),
                          Rescale(min_, max_, -1, 1, transform_skeletons=False),
                          ToTensor()])
    train_dataloader = SequenceDataLoader(train_dataset, BATCH_SIZE,
                                          transforms,
                                          shuffle_sequences=True,
                                          shuffle_frames=True,
                                          device=DEVICE)
    pose_net = PoseResNet(n_channels=3, n_joints=22)
    pose_net.load_state_dict(torch.load(POSE_NET_PATH))
    net = AdaFuse(n_channels=3, n_joints=22, PoseResNet=pose_net, heatmap_dimensions=(512 // 4, 512 // 4))
    net = net.to(DEVICE)

    for data in train_dataloader:
        for key, value in data.items():
            print(key, value.shape)
        with torch.no_grad():
            hm, loss, skely = net(data, False)
            break
        print(skely)
        
