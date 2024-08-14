import torch
from torch import nn

from torchvision.ops import DeformConv2d

from Networks.ResNet import ResNet18FeatureMap, ResNet50FeatureMap
from Networks.UNet import UNetFeatures
from Networks.CenterNetRep.models.networks import pose_dla_dcn
from Networks.CenterNetRep.models.networks import resnet_dcn
#from ResNet import ResNet18FeatureMap
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
        
        x_offs = offset_map[b_idx, 0, h_idx, w_idx] #offset_map[:, 0, None].expand(-1, j, -1, -1)#.contiguous()
        y_offs = offset_map[b_idx, 1, h_idx, w_idx] #offset_map[:, 1, None].expand(-1, j, -1, -1)#.contiguous()
        #relevant_x_offs = x_offs[mask]
        #relevant_y_offs = y_offs[mask]
        #relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 0)
      
        gt_x_offs = gt_offset_map[b_idx, 0, h_idx, w_idx] #gt_offset_map[:, 0, None].expand(-1, j, -1, -1)#.contiguous()
        gt_y_offs = gt_offset_map[b_idx, 1, h_idx, w_idx] #gt_offset_map[:, 1, None].expand(-1, j, -1, -1)#.contiguous()
        #gt_relevant_x_offs = gt_x_offs[mask]
        #gt_relevant_y_offs = gt_y_offs[mask]
        #gt_relevant_offsets = torch.stack((gt_relevant_x_offs, gt_relevant_y_offs), 0)
        return (nn.functional.l1_loss(x_offs, gt_x_offs) + nn.functional.l1_loss(y_offs, gt_y_offs)) / (num + 1e-7) #nn.functional.l1_loss(relevant_offsets, gt_relevant_offsets, size_average=False) / (num + 1e-4)
        
    def depth_loss(self, depth_map, gt_depth_map, mask):
        num = mask.float().sum()*2
        b, j, h, w = mask.shape
        b_idx, j_idx, h_idx, w_idx = torch.where(mask)
        
        #depth_map_repeated = depth_map.expand(-1, j, -1, -1)
        relevant_depths = depth_map[b_idx, 0, h_idx, w_idx] #depth_map_repeated[mask]

        #gt_depth_map_repeated = gt_depth_map.expand(-1, j, -1, -1)
        gt_relevant_depths = gt_depth_map[b_idx, 0, h_idx, w_idx] #gt_depth_map_repeated[mask]
        
        return nn.functional.l1_loss(relevant_depths, gt_relevant_depths, size_average=False) / (num + 1e-7)

    def loss(self, predicted_heat_maps, predicted_offset_maps, predicted_depth_maps,
             gt_heat_maps, gt_offset_maps, gt_depth_maps):
        term1 = self.heatmap_loss(predicted_heat_maps, gt_heat_maps, 4, 2) #nn.functional.cross_entropy(predicted_heat_maps, gt_heat_maps) #self.heatmap_loss(predicted_heat_maps, gt_heat_maps, 4, 2)
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

    
class UpConvolution(nn.Module):
    def __init__(self, input_size, output_size, deformable_kernel_size, kernel_size, stride):
        super().__init__()
        self.deformable_convolution = DeformConv2d(input_size, output_size, kernel_size, 1, 1)
        self.offset_convolution = nn.Conv2d(input_size, 2 * kernel_size * kernel_size, kernel_size, 1, 1)
        self.convolution = nn.Sequential(nn.ConvTranspose2d(output_size, output_size, kernel_size, stride),
                                         nn.BatchNorm2d(output_size),
                                         nn.ReLU())
        
    def forward(self, x):
        offsets = self.offset_convolution(x)
        features = self.deformable_convolution(x, offsets)
        return self.convolution(features)

class CenterNet3DPoseEstimation(nn.Module):
    def __init__(self, n_channels, n_joints):
        super().__init__()
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels 
        self.encoder = ResNet18FeatureMap(n_channels)

        self.decoder = nn.Sequential(UpConvolution(512, 256, 4, 3, 2),
                                     nn.Conv2d(256, 256, 3),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     
                                     UpConvolution(256, 128, 4, 3, 2),
                                     nn.Conv2d(128, 128, 3),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     
                                     UpConvolution(128, 64, 4, 3, 2),
                                     nn.Conv2d(64, 64, 3),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())                                     

        self.key_points_head = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(),
                                             nn.Conv2d(64, n_joints, 1))                                     
        self.offset_head = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 2, 1))
        self.depth_head = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 1, 1),
                                        nn.Sigmoid())

        self.l1_loss = nn.L1Loss()

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features)

        heat_maps = self.key_points_head(features)
        offsets = self.offset_head(features)
        depths = (1 / self.depth_head(features)) - 1

        return heat_maps, offsets, depths

    def heatmap_loss(self, predicted_heat_maps, gt_heat_maps):
        b, c, h, w = predicted_heat_maps.shape
        mask = gt_heat_maps == 1
        opposite_mask = mask.logical_not()

        predicted_heat_maps = torch.sigmoid(predicted_heat_maps)
        positive_predicted_heat_maps = predicted_heat_maps[mask]
        alpha = 2
        true_part = ((1 - positive_predicted_heat_maps) ** alpha) * torch.log(positive_predicted_heat_maps)
        beta = 4
        negative_predicted_heat_maps = predicted_heat_maps[opposite_mask]
        negative_gt_heat_maps = gt_heat_maps[opposite_mask]
        false_part = ((1 - negative_gt_heat_maps) ** beta) * (negative_predicted_heat_maps ** alpha) * torch.log(1 - negative_predicted_heat_maps)
        loss = -1 * (torch.sum(true_part) + torch.sum(false_part))
        return loss / (b * c)

    """def offset_loss(self, predicted_offsets, gt_offsets): 
        return self.l1_loss(predicted_offsets, gt_offsets)
    
    def depth_loss(self, predicted_depths, gt_depths):
        return self.l1_loss(predicted_depths, gt_depths)
    
    def loss(self, predicted_heat_maps, gt_heat_maps, predicted_offsets, gt_offsets, predicted_depths, gt_depths):
        term1 = self.heatmap_loss(predicted_heat_maps, gt_heat_maps)
        #print('heatmap loss')
        #print(term1)

        term2 = self.offset_loss(predicted_offsets, gt_offsets)
        #print('offset loss')
        #print(term2)

        term3 = self.depth_loss(predicted_depths, gt_depths)
        #print('depth loss')
        #print(term3)
        #print()
        return term1 + term2 + term3  """

    def offset_loss(self, predicted_offsets, gt_offsets, gt_heat_maps):
        b, j, h, w = gt_heat_maps.shape
        heat_maps_flattened = torch.flatten(gt_heat_maps, 2)
        idx = torch.argmax(heat_maps_flattened, 2)[:, :, None]

        offsets_flattened = torch.flatten(predicted_offsets, 2)
        x_offs = offsets_flattened[:, 0, None, :].expand(-1, j, -1)
        y_offs = offsets_flattened[:, 1, None, :].expand(-1, j, -1)
        relevant_x_offs = torch.take_along_dim(x_offs, idx, 2).squeeze()
        relevant_y_offs = torch.take_along_dim(y_offs, idx, 2).squeeze()
        relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 1)

        offsets_flattened = torch.flatten(gt_offsets, 2)
        x_offs = offsets_flattened[:, 0, None, :].expand(-1, j, -1)
        y_offs = offsets_flattened[:, 1, None, :].expand(-1, j, -1)
        relevant_x_offs = torch.take_along_dim(x_offs, idx, 2).squeeze()
        relevant_y_offs = torch.take_along_dim(y_offs, idx, 2).squeeze()
        relevant_offsets_gt = torch.stack((relevant_x_offs, relevant_y_offs), 1)
        
        return self.l1_loss(relevant_offsets, relevant_offsets_gt)
    
    def depth_loss(self, predicted_depths, gt_depths, gt_heat_maps):
        b, j, h, w = gt_heat_maps.shape
        heat_maps_flattened = torch.flatten(gt_heat_maps, 2)
        idx = torch.argmax(heat_maps_flattened, 2)[:, :, None]

        depths_flattened = torch.flatten(predicted_depths, 2)
        relevant_depths = torch.take_along_dim(depths_flattened, idx, 2).squeeze()

        depths_flattened = torch.flatten(gt_depths, 2)
        relevant_depths_gt = torch.take_along_dim(depths_flattened, idx, 2).squeeze()
        
        return self.l1_loss(relevant_depths, relevant_depths_gt)
    
    def loss(self, predicted_heat_maps, gt_heat_maps, predicted_offsets, gt_offsets, predicted_depths, gt_depths):
        term1 = self.heatmap_loss(predicted_heat_maps, gt_heat_maps)
        #print('heatmap loss')
        #print(term1)

        term2 = self.offset_loss(predicted_offsets, gt_offsets, gt_heat_maps)
        #print('offset loss')
        #print(term2)

        term3 = self.depth_loss(predicted_depths, gt_depths, gt_heat_maps)
        #print('depth loss')
        #print(term3)
        #print()
        return term1 + term2 + term3   

class CenterNet3DPoseEstimationSoftMax(CenterNet3DPoseEstimation):
    def __init__(self, n_channels, n_joints):
        super().__init__(n_channels, n_joints)

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features)

        heat_maps = self.key_points_head(features)
        exp_ = torch.exp(heat_maps)
        sum_ = torch.sum(exp_, (2, 3))
        heat_maps = exp_ / sum_[:, :, None, None]
        offsets = self.offset_head(features)
        depths = (1 / self.depth_head(features)) - 1

        return heat_maps, offsets, depths

    def heatmap_loss(self, predicted_heat_maps, gt_heat_maps):
        return self.l1_loss(predicted_heat_maps, gt_heat_maps)
    
class CenterNet503DPoseEstimation(CenterNet3DPoseEstimation):
    def __init__(self, n_channels, n_joints):
        super().__init__(n_channels, n_joints)
        self.encoder = ResNet50FeatureMap(n_channels)

        self.decoder = nn.Sequential(UpConvolution(2048, 512, 4, 3, 2),
                                     nn.Conv2d(512, 512, 3),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     
                                     UpConvolution(512, 256, 4, 3, 2),
                                     nn.Conv2d(256, 256, 3),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     
                                     UpConvolution(256, 128, 4, 3, 2),
                                     nn.Conv2d(128, 128, 3),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU())                                     

        self.key_points_head = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(),
                                             nn.Conv2d(64, n_joints, 1))                                     
        self.offset_head = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(),
                                             nn.Conv2d(64, 2, 1))
        self.depth_head = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 1, 1),
                                        nn.Sigmoid())

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

    
    
class CenterNetResNet18(nn.Module):
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

    def heatmap_loss(self, predicted_heat_maps, gt_heat_maps, alpha=2, beta=4):       
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
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    """def offset_loss(self, offset_map, indices_1D, gt_position_2D, stride):
        b, j, hw = indices_1D.shape
        b, _, h, w = offset_map.shape
        offsets_flattened = torch.flatten(offset_map, 2)
        x_offs = offsets_flattened[:, 0, None, :].expand(-1, j, -1)#.contiguous()
        y_offs = offsets_flattened[:, 1, None, :].expand(-1, j, -1)#.contiguous()
        #print(x_offs.is_contiguous())
        #print(indices_1D.is_contiguous())
        relevant_x_offs = torch.take_along_dim(x_offs, indices_1D, 2).squeeze()
        relevant_y_offs = torch.take_along_dim(y_offs, indices_1D, 2).squeeze()
        relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 1)
      
        xs = indices_1D % w
        ys = indices_1D // w
        positions_2D = torch.stack((xs, ys), 1).squeeze()
        gt_offset = (gt_position_2D / stride) - positions_2D
        return nn.functional.l1_loss(relevant_offsets, gt_offset)# self.l1_loss(relevant_offsets, gt_offset)
        
    def depth_loss(self, depth_map, indices_1D, gt_depths):
        depths_flattened = torch.flatten(depth_map, 2)
        relevant_depths = torch.take_along_dim(depths_flattened, indices_1D, 2).squeeze()[:, None, :]
        return nn.functional.l1_loss(relevant_depths, gt_depths) #self.l1_loss(relevant_depths, gt_depths)

    def loss(self, predicted_heat_maps, predicted_offsets, predicted_depths, gt_heat_maps, skeleton_2D_position, skeleton_depths, stride=4):
        term1 = self.heatmap_loss(predicted_heat_maps, gt_heat_maps)
        print('heatmap loss')
        print(term1)

        heat_maps_flattened = torch.flatten(gt_heat_maps, 2)
        idx = torch.argmax(heat_maps_flattened, 2)[:, :, None]
        
        term2 = self.offset_loss(predicted_offsets, idx, skeleton_2D_position, stride)
        print('offset loss')
        print(term2)

        term3 = self.depth_loss(predicted_depths, idx, skeleton_depths)
        print('depth loss')
        print(term3)
        print()
        return term1 + term2 + term3"""

    def offset_loss(self, offset_map, gt_offset_map, mask):
        b, j, h, w = mask.shape
        num = mask.float().sum()*2
        
        x_offs = offset_map[:, 0, None].expand(-1, j, -1, -1)#.contiguous()
        y_offs = offset_map[:, 1, None].expand(-1, j, -1, -1)#.contiguous()
        relevant_x_offs = x_offs[mask]
        relevant_y_offs = y_offs[mask]
        relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 0)
      
        gt_x_offs = gt_offset_map[:, 0, None].expand(-1, j, -1, -1)#.contiguous()
        gt_y_offs = gt_offset_map[:, 1, None].expand(-1, j, -1, -1)#.contiguous()
        gt_relevant_x_offs = gt_x_offs[mask]
        gt_relevant_y_offs = gt_y_offs[mask]
        gt_relevant_offsets = torch.stack((gt_relevant_x_offs, gt_relevant_y_offs), 0)
        return nn.functional.l1_loss(relevant_offsets, gt_relevant_offsets, size_average=False) / (num + 1e-4)
        
    def depth_loss(self, depth_map, gt_depth_map, mask):
        num = mask.float().sum()*2
        b, j, h, w = mask.shape
        
        depth_map_repeated = depth_map.expand(-1, j, -1, -1)
        relevant_depths = depth_map_repeated[mask]

        gt_depth_map_repeated = gt_depth_map.expand(-1, j, -1, -1)
        gt_relevant_depths = gt_depth_map_repeated[mask]
        
        return nn.functional.l1_loss(relevant_depths, gt_relevant_depths, size_average=False) / (num + 1e-4)

    def loss(self, predicted_heat_maps, predicted_offset_maps, predicted_depth_maps,
             gt_heat_maps, gt_offset_maps, gt_depth_maps):
        term1 = self.heatmap_loss(predicted_heat_maps, gt_heat_maps, 4, 2) #nn.functional.cross_entropy(predicted_heat_maps, gt_heat_maps) #self.heatmap_loss(predicted_heat_maps, gt_heat_maps, 4, 2)
        print('heatmap loss')
        print(term1)

        mask = gt_heat_maps.eq(1.0)
        
        term2 = self.offset_loss(predicted_offset_maps, gt_offset_maps, mask)
        print('offset loss')
        print(term2)

        term3 = self.depth_loss(predicted_depth_maps, gt_depth_maps, mask)
        print('depth loss')
        print(term3)
        print()
        return term1 + term2 + term3    

class CenterNetResNet50(nn.Module):
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

    def heatmap_loss(self, predicted_heat_maps, gt_heat_maps, alpha=2, beta=4):       
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
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def offset_loss(self, offset_map, gt_offset_map, mask):
        b, j, h, w = mask.shape
        num = mask.float().sum()*2
        
        x_offs = offset_map[:, 0, None].expand(-1, j, -1, -1)#.contiguous()
        y_offs = offset_map[:, 1, None].expand(-1, j, -1, -1)#.contiguous()
        relevant_x_offs = x_offs[mask]
        relevant_y_offs = y_offs[mask]
        relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 0)
      
        gt_x_offs = gt_offset_map[:, 0, None].expand(-1, j, -1, -1)#.contiguous()
        gt_y_offs = gt_offset_map[:, 1, None].expand(-1, j, -1, -1)#.contiguous()
        gt_relevant_x_offs = gt_x_offs[mask]
        gt_relevant_y_offs = gt_y_offs[mask]
        gt_relevant_offsets = torch.stack((gt_relevant_x_offs, gt_relevant_y_offs), 0)
        return nn.functional.l1_loss(relevant_offsets, gt_relevant_offsets, size_average=False) / (num + 1e-4)
        
    def depth_loss(self, depth_map, gt_depth_map, mask):
        num = mask.float().sum()*2
        b, j, h, w = mask.shape
        
        depth_map_repeated = depth_map.expand(-1, j, -1, -1)
        relevant_depths = depth_map_repeated[mask]

        gt_depth_map_repeated = gt_depth_map.expand(-1, j, -1, -1)
        gt_relevant_depths = gt_depth_map_repeated[mask]
        
        return nn.functional.l1_loss(relevant_depths, gt_relevant_depths, size_average=False) / (num + 1e-4)

    def loss(self, predicted_heat_maps, predicted_offset_maps, predicted_depth_maps,
             gt_heat_maps, gt_offset_maps, gt_depth_maps):
        term1 = self.heatmap_loss(predicted_heat_maps, gt_heat_maps, 4, 2) #nn.functional.cross_entropy(predicted_heat_maps, gt_heat_maps) #self.heatmap_loss(predicted_heat_maps, gt_heat_maps, 4, 2)
        print('heatmap loss')
        print(term1)

        mask = gt_heat_maps.eq(1.0)
        
        term2 = self.offset_loss(predicted_offset_maps, gt_offset_maps, mask)
        print('offset loss')
        print(term2)

        term3 = self.depth_loss(predicted_depth_maps, gt_depth_maps, mask)
        print('depth loss')
        print(term3)
        print()
        return term1 + term2 + term3    
   
"""class CenterNetOld(CenterNet3DPoseEstimation):
    def __init__(self, n_channels, n_joints):
        super().__init__(n_channels, n_joints)

    def offset_loss(self, predicted_offsets, gt_offsets, gt_heat_maps):
        b, j, h, w = gt_heat_maps.shape
        heat_maps_flattened = torch.flatten(gt_heat_maps, 2)
        idx = torch.argmax(heat_maps_flattened, 2)[:, :, None]

        offsets_flattened = torch.flatten(predicted_offsets, 2)
        x_offs = offsets_flattened[:, 0, None, :].expand(-1, j, -1)
        y_offs = offsets_flattened[:, 1, None, :].expand(-1, j, -1)
        relevant_x_offs = torch.take_along_dim(x_offs, idx, 2).squeeze()
        relevant_y_offs = torch.take_along_dim(y_offs, idx, 2).squeeze()
        relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 1)

        offsets_flattened = torch.flatten(gt_offsets, 2)
        x_offs = offsets_flattened[:, 0, None, :].expand(-1, j, -1)
        y_offs = offsets_flattened[:, 1, None, :].expand(-1, j, -1)
        relevant_x_offs = torch.take_along_dim(x_offs, idx, 2).squeeze()
        relevant_y_offs = torch.take_along_dim(y_offs, idx, 2).squeeze()
        relevant_offsets_gt = torch.stack((relevant_x_offs, relevant_y_offs), 1)
        
        return self.l1_loss(relevant_offsets, relevant_offsets_gt)
    
    def depth_loss(self, predicted_depths, gt_depths, gt_heat_maps):
        b, j, h, w = gt_heat_maps.shape
        heat_maps_flattened = torch.flatten(gt_heat_maps, 2)
        idx = torch.argmax(heat_maps_flattened, 2)[:, :, None]

        depths_flattened = torch.flatten(predicted_depths, 2)
        relevant_depths = torch.take_along_dim(depths_flattened, idx, 2).squeeze()

        depths_flattened = torch.flatten(gt_depths, 2)
        relevant_depths_gt = torch.take_along_dim(depths_flattened, idx, 2).squeeze()
        
        return self.l1_loss(relevant_depths, relevant_depths_gt)
    
    def loss(self, predicted_heat_maps, gt_heat_maps, predicted_offsets, gt_offsets, predicted_depths, gt_depths):
        term1 = self.heatmap_loss(predicted_heat_maps, gt_heat_maps)
        #print('heatmap loss')
        #print(term1)

        term2 = self.offset_loss(predicted_offsets, gt_offsets, gt_heat_maps)
        #print('offset loss')
        #print(term2)

        term3 = self.depth_loss(predicted_depths, gt_depths, gt_heat_maps)
        #print('depth loss')
        #print(term3)
        #print()
        return term1 + term2 + term3  """ 

class CenterUNet(nn.Module):
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

    def heatmap_loss(self, predicted_heat_maps, gt_heat_maps, alpha=2, beta=4):       
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
            loss = loss - (pos_loss + neg_loss) / (num_pos + 1e-4)
        return loss

    def offset_loss(self, offset_map, gt_offset_map, mask):
        b, j, h, w = mask.shape
        num = mask.float().sum()*2
        
        x_offs = offset_map[:, 0, None].expand(-1, j, -1, -1)#.contiguous()
        y_offs = offset_map[:, 1, None].expand(-1, j, -1, -1)#.contiguous()
        relevant_x_offs = x_offs[mask]
        relevant_y_offs = y_offs[mask]
        relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 0)
      
        gt_x_offs = gt_offset_map[:, 0, None].expand(-1, j, -1, -1)#.contiguous()
        gt_y_offs = gt_offset_map[:, 1, None].expand(-1, j, -1, -1)#.contiguous()
        gt_relevant_x_offs = gt_x_offs[mask]
        gt_relevant_y_offs = gt_y_offs[mask]
        gt_relevant_offsets = torch.stack((gt_relevant_x_offs, gt_relevant_y_offs), 0)
        return nn.functional.l1_loss(relevant_offsets, gt_relevant_offsets, size_average=False) / (num + 1e-4)
        
    def depth_loss(self, depth_map, gt_depth_map, mask):
        num = mask.float().sum()*2
        b, j, h, w = mask.shape
        
        depth_map_repeated = depth_map.expand(-1, j, -1, -1)
        relevant_depths = depth_map_repeated[mask]

        gt_depth_map_repeated = gt_depth_map.expand(-1, j, -1, -1)
        gt_relevant_depths = gt_depth_map_repeated[mask]
        
        return nn.functional.l1_loss(relevant_depths, gt_relevant_depths, size_average=False) / num

    def loss(self, predicted_heat_maps, predicted_offset_maps, predicted_depth_maps,
             gt_heat_maps, gt_offset_maps, gt_depth_maps):
        term1 = self.heatmap_loss(predicted_heat_maps, gt_heat_maps, 2, 4) #nn.functional.cross_entropy(predicted_heat_maps, gt_heat_maps) #self.heatmap_loss(predicted_heat_maps, gt_heat_maps, 4, 2)
        print('heatmap loss')
        print(term1)

        mask = gt_heat_maps.eq(1.0)
        
        term2 = self.offset_loss(predicted_offset_maps, gt_offset_maps, mask)
        print('offset loss')
        print(term2)

        term3 = self.depth_loss(predicted_depth_maps, gt_depth_maps, mask)
        print('depth loss')
        print(term3)
        print()
        return term1 + term2 + term3  
        
