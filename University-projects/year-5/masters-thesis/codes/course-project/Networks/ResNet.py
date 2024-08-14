# https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py

import torch
import torch.nn as  nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x


"""class BottleneckMod(Bottleneck):
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__(in_channels, out_channels, i_downsample, stride)
        self.relu = nn.LeakyReLU()"""

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm1(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      x += identity
      x = self.relu(x)
      return x

    
"""class BlockMod(Block):
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__(in_channels, out_channels, i_downsample, stride)
        self.relu = nn.LeakyReLU()"""
        

class ResNetFeatureMap(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=3):
        super(ResNetFeatureMap, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x1, x2, x3, x4
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

"""class ResNetModFeatureMap(ResNetFeatureMap):
    def __init__(self, ResBlock, layer_list, num_channels=3):
        super().__init__(ResBlock, layer_list, num_channels)
        self.relu = nn.LeakyReLU()
        self.max_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x"""

def ResNet18FeatureMap(channels=3):
    return ResNetFeatureMap(Block, [2,2,2,2], channels)        

def ResNet34FeatureMap(channels=3):
    return ResNetFeatureMap(Block, [3,4,6,3], channels)

def ResNet50FeatureMap(channels=3):
    return ResNetFeatureMap(Bottleneck, [3,4,6,3], channels)
    
def ResNet101FeatureMap(channels=3):
    return ResNetFeatureMap(Bottleneck, [3,4,23,3], channels)

def ResNet152FeatureMap(channels=3):
    return ResNetFeatureMap(Bottleneck, [3,8,36,3], channels)

__RESNETMODELS = {18 : ResNet18FeatureMap, 34 : ResNet34FeatureMap, 50 : ResNet50FeatureMap, 101 : ResNet101FeatureMap, 152 : ResNet152FeatureMap}

def build_ResNet(in_channels, layers):
    return __RESNETMODELS[layers](in_channels)
    
"""def ResNet18ModFeatureMap(channels=3):
    return ResNetModFeatureMap(BlockMod, [2,2,2,2], channels)        

def ResNet50ModFeatureMap(channels=3):
    return ResNetModFeatureMap(BottleneckMod, [3,4,6,3], channels)"""
