import torch
from torch import nn


### https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x.clone()
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResidualBlock3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock3, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, stride=1, padding=0),
                        nn.BatchNorm2d(out_channels * 4))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x.clone()
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, n_channels, n_joints, layers=(3, 4, 6, 3), block=None):
        super(ResNet, self).__init__()
        self.n_channels = n_channels
        self.n_joints = self.joints = n_joints
        if block is None:
             block = ResidualBlock
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(n_channels, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(12, stride=1)
        #self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512, n_joints * 3)
        
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
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        samples = x.shape[0]
        poses = torch.reshape(x, (samples, 3, self.n_joints))
        return poses

class ResNet50Regression(ResNet):
    def __init__(self, n_channels, n_joints, layers=(3, 4, 6, 3), block=ResidualBlock3):
        super().__init__(n_channels, n_joints, layers, block)
        if block is None:
             block = ResidualBlock3

        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=1)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AvgPool2d(6, stride=1)
        self.fc = nn.Sequential(nn.Linear(2048, 1024), 
                                nn.BatchNorm1d(1024),
                                nn.ReLU(),
                                nn.Dropout(0.25),
                                
                                nn.Linear(1024, 512), 
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Dropout(0.25),

                                nn.Linear(512, n_joints * 3))
        
    def _make_layer(self, block, planes, blocks, stride=1):
        """downsample = None
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

        #return nn.Sequential(*layers)
        """
        input_size = self.inplanes
        mid_size = planes
        output_size = 4 * mid_size
        layers = []
        for i in range(blocks):         
            downsample = None
            if stride != 1 or input_size != output_size:  
                downsample = nn.Sequential(
                    nn.Conv2d(input_size, output_size, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(output_size),
                )
            layers.append(block(input_size, mid_size, stride, downsample))
            input_size = output_size

        pool = nn.MaxPool2d(3, 2, 1)
        layers.append(pool)
        self.inplanes = input_size
        return nn.Sequential(*layers)

class ResNet50(nn.Module):
    def __init__(self, n_channels, layers=(3, 4, 6, 3), block=ResidualBlock3):
        super().__init__()
        if block is None:
             block = ResidualBlock3

        self.layer1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self._channels = 64
        self.layer2 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer3 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[3], stride=2)

        
    def _make_layer(self, block, filters, blocks, stride=1):
        input_size = self._channels
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

        self._channels = output_size
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class ResNet50Features(ResNet50):
    def __init__(self, n_channels, layers=(3, 4, 6, 3), block=ResidualBlock3):
        super().__init__(n_channels, layers, block)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        return x1, x2, x3, x4, x5
    
class ResNet18(nn.Module):
    def __init__(self, n_channels, layers=(3, 4, 6, 3), block=None):
        super(ResNet18, self).__init__()
        self.n_channels = n_channels
        if block is None:
             block = ResidualBlock
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(n_channels, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        
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
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
    
class ResNet18Features(ResNet18):
    def __init__(self, n_channels, layers=(3, 4, 6, 3), block=None):
        super().__init__(n_channels, layers, block)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.layer0(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        return x1, x2, x3, x4, x5

class ResNet18ModFeatures(ResNet18):
    def __init__(self, n_channels, layers=(3, 4, 6, 3), block=None):
        super().__init__(n_channels, layers, block)
        self.conv1 = nn.Sequential(
                        nn.Conv2d(n_channels, 64, kernel_size=4, stride=1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.layer0(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)
        return x1, x2, x3, x4, x5

if __name__ == '__main__':
    A = torch.rand(16, 3, 360, 360)
    model = ResNet18ModFeatures(3)
    features = model(A)
    for i in range(len(features)):
        print(features[i].shape)