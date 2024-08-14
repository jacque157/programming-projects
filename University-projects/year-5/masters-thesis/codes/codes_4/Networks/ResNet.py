import torch
import torch.nn as  nn
import torch.nn.functional as F


class ResNetBase(nn.Module):
    def __init__(self, n_channels, n_joints, pretrained=True):
        super().__init__()
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels
        self.network = None
        self.regression = None

    def forward(self, x):
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)

        x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)

        x = torch.flatten(x, 1)
        b, _ = x.shape
        x = self.regression(x)

        return torch.reshape(x, (b, 3, self.n_joints))

    def loss(self, predicted_skeletons, gt_skeletons):
        return F.mse_loss(predicted_skeletons, gt_skeletons)

class ResNet18(ResNetBase):
    def __init__(self, n_channels, n_joints, pretrained=True):
        super().__init__(n_channels, n_joints, pretrained)
        self.network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
        if n_channels != 3:
            self.network.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.regression = nn.Sequential(nn.Linear(16*16*512, 1024),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(),

                                        nn.Linear(1024, n_joints * 3))
        del self.network.avgpool
        del self.network.fc
        
class ResNet34(ResNetBase):
    def __init__(self, n_channels, n_joints, pretrained=True):
        super().__init__(n_channels, n_joints, pretrained)
        self.network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=pretrained)
        if n_channels != 3:
            self.network.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.regression = nn.Sequential(nn.Linear(16*16*512, 1024),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(),

                                        nn.Linear(1024, n_joints * 3))
        del self.network.avgpool
        del self.network.fc

class ResNet50(ResNetBase):
    def __init__(self, n_channels, n_joints, pretrained=True):
        super().__init__(n_channels, n_joints, pretrained)
        self.network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
        if n_channels != 3:
            self.network.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.regression = nn.Sequential(nn.Linear(16*16*2048, 2048),
                                        nn.BatchNorm1d(2048),
                                        nn.ReLU(),

                                        nn.Linear(2048, n_joints * 3))
        del self.network.avgpool
        del self.network.fc

class ResNet101(ResNetBase):
    def __init__(self, n_channels, n_joints, pretrained=True):
        super().__init__(n_channels, n_joints, pretrained)
        self.network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=pretrained)
        if n_channels != 3:
            self.network.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.regression = nn.Sequential(nn.Linear(16*16*2048, 2048),
                                        nn.BatchNorm1d(2048),
                                        nn.ReLU(),

                                        nn.Linear(2048, n_joints * 3))
        del self.network.avgpool
        del self.network.fc

class ResNet152(ResNetBase):
    def __init__(self, n_channels, n_joints, pretrained=True):
        super().__init__(n_channels, n_joints, pretrained)
        self.network = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=pretrained)
        if n_channels != 3:
            self.network.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.regression = nn.Sequential(nn.Linear(16*16*2048, 2048),
                                        nn.BatchNorm1d(2048),
                                        nn.ReLU(),

                                        nn.Linear(2048, n_joints * 3))
        del self.network.avgpool
        del self.network.fc

__RESNETMODELS = {18 : ResNet18, 34 : ResNet34, 50 : ResNet50, 101 : ResNet101, 152 : ResNet152}

def build_ResNet(layers, in_channels, n_joints, pretrained=True):
    return __RESNETMODELS[layers](in_channels, n_joints, pretrained)

if __name__ == '__main__':
    channels = 5
    img = torch.rand(16, channels, 512, 512)
    model = build_ResNet(18, channels, 22)
    out = model(img)
