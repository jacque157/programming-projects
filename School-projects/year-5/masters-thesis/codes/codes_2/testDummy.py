import torch

from Networks import PoseEstimation3DNetwork
from Networks import UNet
from Networks import ResNet
from Networks import PoseNet
from Dataset import *
from DatasetLoader import *
from Transforms import *
from utils import *

CHANNELS = 3
model_path = 'models/ResNet50absolutePose_batch=8_dummy_L1_shuffle_lr=0.01/net_20.pt'


def rescale_pointcloud(pt_cloud, mean, min_, max_, a=-1, b=1):
    pt_cloud = (pt_cloud - a) * (max_ - min_) / (b - a)
    pt_cloud += min_
    pt_cloud += mean
    return pt_cloud

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training on {device}')
    min_, max_ = np.load(os.path.join('Dataset', 'min_max.npy'))# find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    mean = np.load(os.path.join('Dataset', 'mean.npy'))
    transforms = transforms.Compose([ZeroCenter(mean),
                                     Rescale(min_, max_, -1, 1),
                                     #RootAlignedPose(),
                                     #Shuffle(),
                                     ToTensor()])
    dummy_dataset = DummyPose3D('Dataset',
                                'CMU',
                                transform=transforms)
    dummy_dataloader = DatasetLoader([dummy_dataset],
                                     8, ToDevice(device))

    model = PoseNet.ResNet50RelativePose(CHANNELS, dummy_dataset.joints)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    for batch in dummy_dataloader:
        seq = batch['sequences']
        skeletons = batch['key_points']
        predictions = model(seq)
        roots = batch['root_key_points']
        
        seq = np.moveaxis(seq.detach().cpu().numpy(), 1, 3)
        skeletons = np.moveaxis(skeletons.detach().cpu().numpy(), 1, 2)
        predictions = np.moveaxis(predictions.detach().cpu().numpy(), 1, 2)
        roots = roots.detach().cpu().numpy()

        for point_cloud, pr_skeleton, skeleton, root in zip(seq, predictions, skeletons, roots):
            #point_cloud = rescale_pointcloud(point_cloud, mean, min_, max_)
            #pr_skeleton = rescale_pointcloud(pr_skeleton, mean, min_, max_)
            #skeleton = rescale_pointcloud(skeleton, mean, min_, max_)
            
            #pr_skeleton = np.concatenate((np.zeros((1, 3)), pr_skeleton), 0)
            #pr_skeleton += root
            ax = plot_body(point_cloud)
            plot_skeleton(pr_skeleton, ax)

            #skeleton += root
            ax = plot_body(point_cloud)
            plot_skeleton(skeleton, ax)

            plt.show()
