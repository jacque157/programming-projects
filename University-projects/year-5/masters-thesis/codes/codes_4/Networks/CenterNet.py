import torch
from torch import nn

#from Networks.UNet import UNetFeatures
from Networks.CenterNetRep.models.networks import pose_dla_dcn
from Networks.CenterNetRep.models.networks import resnet_dcn

from time import time

class CenterNetBase(nn.Module):
    def __init__(self):
        super().__init__()
        
    def heatmap_loss(self, predicted_heat_maps, gt_heat_maps, alpha=2, beta=4):
        predicted_heat_maps = torch.clamp(predicted_heat_maps, min=1e-8, max=1-1e-8)
        pos_inds = gt_heat_maps.eq(1).float()
        neg_inds = gt_heat_maps.lt(1).float()
        neg_weights = torch.pow(1 - gt_heat_maps, beta)

        loss = 0

        pos_loss = torch.log(predicted_heat_maps) * torch.pow(1 - predicted_heat_maps, alpha) * pos_inds
        neg_loss = torch.log(1 - predicted_heat_maps) * torch.pow(predicted_heat_maps, alpha) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / (num_pos + 1e-3)
        return loss

    def offset_loss(self, offset_map, gt_offset_map, mask):
        b, j, h, w = mask.shape
        num = mask.float().sum()*2
        b_idx, j_idx, h_idx, w_idx = torch.where(mask)
        
        x_offs = offset_map[b_idx, 0, h_idx, w_idx] 
        y_offs = offset_map[b_idx, 1, h_idx, w_idx] 
      
        gt_x_offs = gt_offset_map[b_idx, 0, h_idx, w_idx]
        gt_y_offs = gt_offset_map[b_idx, 1, h_idx, w_idx] 
        return (nn.functional.l1_loss(x_offs, gt_x_offs) + nn.functional.l1_loss(y_offs, gt_y_offs)) / (num + 1e-7) #nn.functional.l1_loss(relevant_offsets, gt_relevant_offsets, size_average=False) / (num + 1e-4)
        
    def depth_loss(self, depth_map, gt_depth_map, mask):
        num = mask.float().sum()*2
        b, j, h, w = mask.shape
        b_idx, j_idx, h_idx, w_idx = torch.where(mask)
        
        relevant_depths = depth_map[b_idx, 0, h_idx, w_idx] 
        gt_relevant_depths = gt_depth_map[b_idx, 0, h_idx, w_idx]
        return nn.functional.l1_loss(relevant_depths, gt_relevant_depths, size_average=False) / (num + 1e-7)

    def loss(self, predicted_heat_maps, predicted_offset_maps, predicted_depth_maps,
             gt_heat_maps, gt_offset_maps, gt_depth_maps):
        term1 = self.heatmap_loss(predicted_heat_maps, gt_heat_maps, 4, 2)#1, 1)#2, 4) #4, 2 #nn.functional.cross_entropy(predicted_heat_maps, gt_heat_maps) #self.heatmap_loss(predicted_heat_maps, gt_heat_maps, 4, 2)
        #print('heatmap loss')
        #print(term1)

        mask = gt_heat_maps.eq(1.0)

        term2 = self.offset_loss(predicted_offset_maps, gt_offset_maps, mask)
        #print('offset loss')
        #print(term2)

        term3 = self.depth_loss(predicted_depth_maps, gt_depth_maps, mask)
        #print('depth loss')
        #print(term3)
        #print()
        return term1 + term2 + term3

class CenterNetDLA(CenterNetBase):
    def __init__(self, n_channels, n_joints):
        super().__init__()
        self.network = pose_dla_dcn.get_pose_net(num_layers=34, heads={'hm' : n_joints, 'dep' : 1, 'reg' : 2}, head_conv=256, down_ratio=4)
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels 
        if n_channels != 3:
            self.network.base.base_layer[0] = nn.Conv2d(n_channels, self.network.base.channels[0],
                                                        kernel_size=7, stride=1,
                                                        padding=3, bias=False)

        
    def forward(self, x):
        result = self.network(x)[0]
        heat_maps = nn.functional.sigmoid(result['hm']) #torch.clamp(self.sigmoid(result['hm']), 1e-4, 1 - 1e-4)
        depths = result['dep']
        depths = (1 / nn.functional.sigmoid(depths)) - 1 
        offsets = nn.functional.sigmoid(result['reg']) 
        return heat_maps, offsets, depths

    
    
class CenterNetResNet18(CenterNetBase):
    def __init__(self, n_channels, n_joints):
        super().__init__()
        self.network = resnet_dcn.get_pose_net(18, {'hm' : n_joints, 'dep' : 1, 'reg' : 2}, 64) # 256 better
        #self.sigmoid = nn.Sigmoid()
        #self.l1_loss = nn.L1Loss()
        
    def forward(self, x):
        result = self.network(x)[0]
        heat_maps = nn.functional.sigmoid(result['hm']) #torch.clamp(self.sigmoid(result['hm']), 1e-4, 1 - 1e-4)
        depths = result['dep']
        depths = (1 / nn.functional.sigmoid(depths)) - 1 
        offsets = nn.functional.sigmoid(result['reg']) 
        return heat_maps, offsets, depths

class CenterNetResNet50(CenterNetBase):
    def __init__(self, n_channels, n_joints):
        super().__init__()
        self.network = resnet_dcn.get_pose_net(50, {'hm' : n_joints, 'dep' : 1, 'reg' : 2}, 64)
        #self.sigmoid = nn.Sigmoid()
        #self.l1_loss = nn.L1Loss()
        
    def forward(self, x):
        result = self.network(x)[0]
        heat_maps = nn.functional.sigmoid(result['hm']) #torch.clamp(self.sigmoid(result['hm']), 1e-4, 1 - 1e-4)
        depths = result['dep']
        depths = (1 / nn.functional.sigmoid(depths)) - 1 
        offsets = nn.functional.sigmoid(result['reg'])
        return heat_maps, offsets, depths
"""
class CenterUNet(CenterNetBase):
    class Head(nn.Module):
        def __init__(self, in_channels, mid_channels, out_channels):
            super().__init__()
            self.net = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, 1, 1),
                                     #nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),

                                     nn.Conv2d(mid_channels, out_channels, 1, 1),)
                                     #nn.BatchNorm2d(out_channels),)
        def forward(self, x):
            return self.net(x)
    
    def __init__(self, n_channels, n_joints, n_layers):
        super().__init__()
        self.u_net = UNetFeatures(n_channels, n_layers)
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels

        out_channels = self.u_net.out_channels

        self.heat_map_head = self.Head(out_channels, 64, n_joints)
        self.offset_head = self.Head(out_channels, 64, 2)
        self.depth_head = self.Head(out_channels, 64, 1)
                    
    def forward(self, x):
        features = self.u_net(x)
        heat_maps = nn.functional.sigmoid(self.heat_map_head(features)) #torch.clamp(self.sigmoid(result['hm']), 1e-4, 1 - 1e-4)
        depths = (1 / nn.functional.sigmoid(self.depth_head(features))) - 1
        offsets = nn.functional.sigmoid(self.offset_head(features))
        return heat_maps, offsets, depths
"""
        
