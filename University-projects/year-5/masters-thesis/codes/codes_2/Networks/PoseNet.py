import torch
from torch import nn

from Networks.ResNet import ResNet50Mod, ResNet18Mod, ResNet18NoBatchNorm


class PoseResNet18(nn.Module):
    def __init__(self, n_channels, n_joints) -> None:
        super().__init__()
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels 
        self.encoder = ResNet18Mod(1024, n_channels)
        self.fc = nn.Sequential(nn.BatchNorm1d(1024),
                                nn.LeakyReLU(),
                                nn.Linear(1024, n_joints * 3))       
    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.encoder(x)
        x = self.fc(x)
        return torch.reshape(x, (B, 3, self.n_joints))
    
class PoseResNet50(nn.Module):
    def __init__(self, n_channels, n_joints) -> None:
        super().__init__()
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels 
        self.encoder = ResNet50Mod(1024, n_channels)
        self.fc = nn.Sequential(nn.BatchNorm1d(1024),
                                nn.LeakyReLU(),
                                nn.Linear(1024, n_joints * 3))       
    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.encoder(x)
        x = self.fc(x)
        return torch.reshape(x, (B, 3, self.n_joints))


class PoseResNet18NoBatchNorm(PoseResNet18):
    def __init__(self, n_channels, n_joints) -> None:
        super().__init__(n_channels, n_joints)
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels 
        self.encoder = ResNet18NoBatchNorm(1024, n_channels)
        self.fc = nn.Sequential(nn.LeakyReLU(),
                                nn.Linear(1024, n_joints * 3))
        


    