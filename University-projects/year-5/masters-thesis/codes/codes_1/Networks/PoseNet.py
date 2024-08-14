import torch
from torch import nn
#from ResNet import ResNet50
from Networks.ResNet import ResNet50Features, ResNet18Features, ResNet18ModFeatures, ResNet18, ResNet50, ResidualBlock, ResidualBlock3
import time

class UpScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                                 nn.Conv2d(out_channels, out_channels,
                                               kernel_size=3, padding=1),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU())
    def forward(self, x):
        return self.net(x)

class OutLayer(nn.Module):
    def __init__(self, n_channels, map_size, n_joints) -> None:
        super().__init__()
        self.net = nn.Conv2d(n_channels, map_size * n_joints, 1, 1)
        self.n_joints = n_joints

    def forward(self, x):
        x = self.net(x)
        b, c, h, w = x.shape
        return x.view(b, self.n_joints, c // self.n_joints, h, w)

class PoseNet50(nn.Module):
    def __init__(self, n_channels, n_joints, device='cuda', heat_map_size=90) -> None:
        super().__init__()
        self.n_joints = self.joints = n_joints
        self.encoder = ResNet50Features(n_channels)

        self.decoder_layer_1 = UpScaleConv(2048, 1024, 3, 2, 1)
        self.decoder_layer_2 = UpScaleConv(1024, 512, 3, 2, 1)
        self.decoder_layer_3 = UpScaleConv(512, 256, 4, 2, 1)
        self.out_layer = OutLayer(256, heat_map_size, n_joints - 1)

        self.root_regression = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                             nn.Flatten(),

                                             nn.Linear(2048, 1024), 
                                             nn.BatchNorm1d(1024),
                                             nn.ReLU(),
                                             nn.Dropout(0.5),
                                             
                                             nn.Linear(1024, 512), 
                                             nn.BatchNorm1d(512),
                                             nn.ReLU(),
                                             nn.Dropout(0.5),

                                             nn.Linear(512, 3))
        
        self.points = self._init_points(heat_map_size, device)

    def _init_points(self, heat_map_size, device):
        """Creates volume where each voxel represents point in space"""
        H, W, D = heat_map_size, heat_map_size, heat_map_size
        numbers = torch.arange(H * W * D).reshape(H, W, D)
        indices = torch.zeros(3, H, W, D)
        indices[2, :, :, :] = numbers % D
        indices[1, :, :, :] = (numbers // D) % W
        indices[0, :, :, :] = ((numbers // D) // W) % H

        unit_size = 1210 / (heat_map_size / 2) # asuming that distance between root joint and any other joint is no more than 1210 mm
        points = indices - (heat_map_size / 2)
        points *= unit_size 
        return points.to(device).requires_grad_(False)
                      
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)

        x = self.decoder_layer_1(x5)
        x = self.decoder_layer_2(x + x4)
        x = self.decoder_layer_3(x + x3)

        x = self.out_layer(x + x2)

        joints = reconstruct_pose(self.points, x)
        root_joint = self.root_regression(x5)
        return root_joint, joints

class PoseNet18(PoseNet50):
    def __init__(self, n_channels, n_joints, device='cuda', heat_map_size=90) -> None:
        super().__init__(n_channels, n_joints, device, heat_map_size)
        self.encoder = ResNet18Features(n_channels)

        self.decoder_layer_1 = UpScaleConv(512, 256, 3, 2, 1)
        self.decoder_layer_2 = UpScaleConv(256, 128, 3, 2, 1)
        self.decoder_layer_3 = UpScaleConv(128, 64, 4, 2, 1)
        self.out_layer = OutLayer(64, heat_map_size, n_joints - 1)

        self.root_regression = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                             nn.Flatten(),

                                             nn.Linear(512, 256), 
                                             nn.BatchNorm1d(256),
                                             nn.ReLU(),
                                             nn.Dropout(0.5),
                                             
                                             nn.Linear(256, 128), 
                                             nn.BatchNorm1d(128),
                                             nn.ReLU(),
                                             nn.Dropout(0.5),

                                             nn.Linear(128, 3))
        
class RelativePoseNet18(nn.Module):
    def __init__(self, n_channels, n_joints, device='cuda', heat_map_size=180) -> None:
        super().__init__()
        self.n_joints = self.joints = n_joints
        self.encoder = ResNet18ModFeatures(n_channels)

        self.decoder_layer_1 = UpScaleConv(512, 256, 3, 2, 1)
        self.decoder_layer_2 = UpScaleConv(256, 128, 4, 2, 1)
        self.decoder_layer_3 = UpScaleConv(128, 64, 4, 2, 1)
        self.out_layer = OutLayer(64, heat_map_size, n_joints) 
        
        self.points = self._init_points(heat_map_size, device)

    def _init_points(self, heat_map_size, device):
        """Creates volume where each voxel represents point in space"""
        H, W, D = heat_map_size, heat_map_size, heat_map_size
        numbers = torch.arange(H * W * D).reshape(H, W, D)
        indices = torch.zeros(3, H, W, D)
        indices[2, :, :, :] = numbers % D
        indices[1, :, :, :] = (numbers // D) % W
        indices[0, :, :, :] = ((numbers // D) // W) % H

        unit_size = 1210 / (heat_map_size / 2) # asuming that distance between root joint and any other joint is no more than 1210 mm
        points = indices - (heat_map_size / 2)
        points *= unit_size 
        return points.to(device).requires_grad_(False)
                      
    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)

        x = self.decoder_layer_1(x5)
        x = self.decoder_layer_2(x + x4)
        x = self.decoder_layer_3(x + x3)

        x = self.out_layer(x + x2)
        joints = reconstruct_pose(self.points, x)
        return joints

class RelativePoseNet50(RelativePoseNet18):
    def __init__(self, n_channels, n_joints, device='cuda', heat_map_size=90) -> None:
        super().__init__(n_channels, n_joints, device, heat_map_size)
        self.n_joints = self.joints = n_joints
        self.encoder = ResNet50Features(n_channels)

        self.decoder_layer_1 = UpScaleConv(2048, 1024, 3, 2, 1)
        self.decoder_layer_2 = UpScaleConv(1024, 512, 3, 2, 1)
        self.decoder_layer_3 = UpScaleConv(512, 256, 4, 2, 1)
        self.out_layer = OutLayer(256, heat_map_size, n_joints) 
        
        self.points = self._init_points(heat_map_size, device)

class PoseNet18Regression(nn.Module):
    def __init__(self, in_channels, n_joints) -> None:
        super().__init__()
        self.joints = self.n_joints = n_joints
        self.encoder = ResNet18(in_channels)
        self.regression = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                        nn.Flatten(),
                                        nn.Linear(512, 256), 
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        
                                        nn.Linear(256, 128), 
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.Linear(128, 3 * n_joints))
        
    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.encoder(x)
        x = self.regression(x)
        return torch.reshape(x, (B, 3, self.n_joints))
    
class PoseNet50Regression(PoseNet18Regression):
    def __init__(self, in_channels, n_joints) -> None:
        super().__init__(in_channels, n_joints)
        self.encoder = ResNet50(in_channels)
        self.regression = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                        nn.Flatten(),
                                        nn.Linear(2048, 1024), 
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        
                                        nn.Linear(1024, 512), 
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.Linear(512, 3 * n_joints))

class PoseResNet(nn.Module):
    def __init__(self, n_channels, n_joints, layers=(3, 2, 3, 2), block=None) -> None:
        super().__init__()
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels
        if block is None:
             block = ResidualBlock
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        
        self.layer0 = self._make_layer(block, 256, layers[0], stride = 2)
        self.layer1 = self._make_layer(block, 512, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 1024, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 2048, layers[3], stride = 2)

        self.regression = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                        nn.Flatten(),
                                        nn.Linear(2048, 1024), 
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(),
                                        nn.Linear(1024, 3 * n_joints),
                                        nn.Tanh())
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.conv1(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.regression(x)
        return torch.reshape(x, (B, 3, self.n_joints))


class ResNet18RelativePose(nn.Module):
    def __init__(self, n_channels, n_joints) -> None:
        super().__init__()
        self.joints = self.n_joints = n_joints
        self.encoder = ResNet18(n_channels)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                nn.Flatten(),
                                nn.Linear(512, 256), 
                                nn.BatchNorm1d(256),
                                nn.ReLU(),
                                nn.Linear(256, 3 * n_joints),
                                nn.Tanh())
        
    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.encoder(x)
        x = self.fc(x)
        return torch.reshape(x, (B, 3, self.n_joints))


class ResNet50RelativePose(nn.Module):
    def __init__(self, n_channels, n_joints) -> None:
        super().__init__()
        self.joints = self.n_joints = n_joints
        self.encoder = ResNet50(n_channels)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                nn.Flatten(),
                                nn.Linear(2048, 1024), 
                                nn.BatchNorm1d(1024),
                                nn.ReLU(),
                                nn.Linear(1024, 3 * n_joints),
                                nn.Tanh())
        
    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.encoder(x)
        print(x.shape)
        x = self.fc(x)
        return torch.reshape(x, (B, 3, self.n_joints))

class PoseResNet50(PoseResNet):
    def __init__(self, n_channels, n_joints, layers=(3, 3, 3, 3), block=None) -> None:
        super().__init__(n_channels, n_joints, layers, block)
        if block is None:
             block = ResidualBlock3

        self.n_channels = 64
        self.layer0 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)

        self.regression = nn.Sequential(nn.AdaptiveAvgPool2d(1), 
                                        nn.Flatten(),
                                        nn.Linear(2048, 512), 
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        
                                        nn.Linear(512, 256), 
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.Linear(256, 3 * n_joints),
                                        nn.Tanh())

        
    def _make_layer(self, block, filters, blocks, stride=1):
        input_size = self.n_channels
        output_size = 4 * filters
        layers = []
        for i in range(blocks):       
            downsample = None
            if (stride != 1 and i + 1 == blocks) :  
                downsample = nn.Sequential(
                    nn.Conv2d(input_size, output_size, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(output_size),
                )
            elif input_size != output_size:
                downsample = nn.Sequential(
                    nn.Conv2d(input_size, output_size, kernel_size=1, stride=1),
                    nn.BatchNorm2d(output_size),
                )
            if i + 1 == blocks:
                layers.append(block(input_size, filters, stride, downsample))
            else:
                layers.append(block(input_size, filters, 1, downsample))
            input_size = output_size

        self.n_channels = output_size
        return nn.Sequential(*layers)

def reconstruct_pose(points, features):
    """Performs arg-softmax on points using heat maps generated from features"""
    N, C, H, W, D = features.shape
    start = time.time()
    soft_max = nn.functional.softmax(features.reshape(N, C, -1), dim=2)
    print(f'Time taken for softmax: {time.time() - start}')
    start = time.time()
    points = torch.sum(points.view(1, 1, 3, -1) * soft_max.view(N, C, 1, -1), dim=3)
    print(f'Time taken for arg-softmax: {time.time() - start}')
    start = time.time()
    #points = soft_argmax(features)
    #print(f'Time taken for arg-softmax: {time.time() - start}')
    start = time.time()
    points = torch.moveaxis(points, 2, 1)
    print(f'Time taken for moveaxis: {time.time() - start}')
    return points

def soft_argmax(voxels):
    # https://github.com/Fdevmsy/PyTorch-Soft-Argmax/blob/master/soft-argmax.py
    """
    Arguments: voxel patch in shape (batch_size, channel, H, W, depth)
    Return: 3D coordinates in shape (batch_size, channel, 3)
    """
    assert voxels.dim()==5
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1#1000.0 
    N,C,H,W,D = voxels.shape
    soft_max = nn.functional.softmax(voxels.reshape(N,C,-1)*alpha,dim=2)
    soft_max = soft_max.view(voxels.shape)
    indices_kernel = torch.arange(start=0,end=H*W*D).unsqueeze(0).to('cuda')
    indices_kernel = indices_kernel.view((H,W,D))
    conv = soft_max*indices_kernel
    indices = conv.sum(2).sum(2).sum(2)
    z = indices%D
    y = (indices/D).floor()%W
    x = (((indices/D).floor())/W).floor()%H
    coords = torch.stack([x,y,z],dim=2)
    return coords