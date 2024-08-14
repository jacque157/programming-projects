#from networks import resnet_dcn
#import CenterNetRep
from CenterNetRep.models.networks import resnet_dcn
from CenterNetRep.models.networks import pose_dla_dcn
import torch


net = resnet_dcn.get_pose_net(18, {'hm' : 24, 'dep' : 1, 'reg' : 2}, 64)
a = torch.rand(32, 3, 256, 128)
b = net(a)

net = pose_dla_dcn.get_pose_net(num_layers=34, heads={'hm' : 22, 'dep' : 1, 'reg' : 2}, head_conv=256, down_ratio=4)
a = torch.rand(32, 3, 256, 128)
b = net(a)
