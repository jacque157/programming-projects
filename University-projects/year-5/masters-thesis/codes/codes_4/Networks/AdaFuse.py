import torch
from torch import nn

from Networks.AdaFuseRep.models.pose_resnet import get_pose_net
from Networks.AdaFuseRep.models.adafuse_network import get_multiview_pose_net
from Networks.AdaFuseRep.core.loss import JointsMSELoss#, JointMPJPELoss
#from AdaFuseRep.models.pose_resnet import get_pose_net
#from AdaFuseRep.models.adafuse_network import get_multiview_pose_net
#from AdaFuseRep.core.loss import JointsMSELoss
from easydict import EasyDict as edict


def get_fake_config():
    config = edict()
    config.POSE_RESNET = edict()
    config.POSE_RESNET.NUM_LAYERS = 50
    config.POSE_RESNET.DECONV_WITH_BIAS = False
    config.POSE_RESNET.NUM_DECONV_LAYERS = 3
    config.POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
    config.POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
    config.POSE_RESNET.FINAL_CONV_KERNEL = 1

    config.NETWORK = edict()
    config.NETWORK.PRETRAINED = 'pretrained/pose_resnet_50_256x192.pth.tar'
    config.NETWORK.NUM_JOINTS = None
    config.NETWORK.HEATMAP_SIZE = None
    config.NETWORK.SIGMA = 2
    
    config.DATASET = edict()
    config.DATASET.NUM_USED_JOINTS = None
    config.DATASET.TRAIN_DATASET = 'panoptic'
    
    config.CAM_FUSION = edict()
    config.CAM_FUSION.CROSSVIEW_FUSION = True
    
    config.MULTI_CAMS = edict()
    config.MULTI_CAMS.SELECTED_CAMS = [0, 1, 2, 3]
    return config

def get_fake_mapping(n_joints):
    return torch.arange(n_joints)

class PoseResNet(nn.Module):
    def __init__(self, n_channels, n_joints, pretrained=False):
        super().__init__()
        config_hack = get_fake_config()
        config_hack.NETWORK.NUM_JOINTS = n_joints
        config_hack.NETWORK.NUM_USED_JOINTS = n_joints
        
        self.network = get_pose_net(config_hack, is_train=pretrained)
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels
        if n_channels != 3:
            self.network.conv1 = nn.Conv2d(n_channels, 64,
                                           kernel_size=7, stride=2,
                                           padding=3, bias=False)
        self.loss_fun = JointsMSELoss(use_target_weight=False)

    def forward(self, x):
        return self.network(x)

    def loss(self, predicted_heat_map, gt_heat_map):
        return self.loss_fun(predicted_heat_map, gt_heat_map, None)
        
class AdaFuse(nn.Module):
    def __init__(self, n_channels, n_joints, PoseResNet, heatmap_dimensions):
        super().__init__()
        config_hack = get_fake_config()
        config_hack.NETWORK.NUM_JOINTS = n_joints
        config_hack.NETWORK.NUM_USED_JOINTS = n_joints
        config_hack.NETWORK.HEATMAP_SIZE = heatmap_dimensions
        config_hack.DATASET.NUM_USED_JOINTS = n_joints
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels
        self.network = get_multiview_pose_net(PoseResNet, config_hack)
        self.network.joint_mapping = get_fake_mapping(n_joints)
        #self.loss_fun  = JointsMSELoss(use_target_weight=True)

    def forward(self, data, train=True):
        #R = data['extrinsic_matrix'][0:3, 0:3]
        #t = data['extrinsic_matrix'][0:3, 3]
        x = data['point_clouds']
        b, v, c, h, w = x.shape
        self.network.b_ransac = False if train else True
        params = {'run_phase' : 'train' if train else 'test',
                  'camera_R' : data['camera_rotations'][None].expand(b, -1, -1, -1),#.contiguous(),
                  'camera_T': data['camera_positions'][None, :, :, None].expand(b, -1, -1, -1),#.contiguous(),
                  'camera_Intri' : data['intrinsic_matrix'][None, None].expand(b, v, -1, -1),#.contiguous(),
                  'aug_trans' : torch.eye(3).to(x.device)[None, None].expand(b, v, -1, -1),#.contiguous(),
                  'joint_vis' : [],
                  'joints_gt' : torch.moveaxis(data['skeletons'], 1, -1)[:, None].expand(-1, v, -1, -1),#.contiguous(),
                  'joints_vis' : data['visible_joints'][:, :, :, None]}
        hms_out, out_data =  self.network(x, **params)
        if 'j3d_ransac' in out_data:
            predicted_skeletons = torch.moveaxis(out_data['j3d_ransac'], -1, 1)
        else:
            predicted_skeletons = None
        return hms_out, out_data['joint_2d_loss'].mean(), predicted_skeletons

    """def loss(self, predicted_heatmaps, predicted_weights joint_2d_loss)
            loss = 0
                if train_2d_backbone:
                    loss_mse = criterion_mse(hms, target_cuda, weight_cuda)
                    loss += loss_mse
                loss += joint_2d_loss"""
    
if __name__ == '__main__':
    img = torch.rand(32, 4, 512, 512)
    net = PoseResNet(4, 24)
    out = net(img)[0]
    b, j, h, w = out.shape  
    l = net.loss(out, out)
    print(l)
    
    net2 = AdaFuse(4, 24, net, (h, w))
    
