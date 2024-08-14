import torch
from torch import nn 


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, f1, f2, f3, f4, f5, f6) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, f3, 1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, f1, 1), nn.ReLU(),
                                   nn.Conv2d(f1, f4, 3, 1, 1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, f2, 1), nn.ReLU(),
                                   nn.Conv2d(f2, f5, 5, 1, 2), nn.ReLU())
        self.conv4 = nn.Sequential(nn.MaxPool2d(3, 1, 1),
                                   nn.Conv2d(in_channels, f6, 1), nn.ReLU())

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y4 = self.conv4(x)
        y = torch.cat((y1, y2, y3, y4), 1)
        return y 

class GooglePoseNet(nn.Module):
    def __init__(self, in_channels, n_joints) -> None:
        super().__init__()
        self.joints = self.n_joints = n_joints
        self.in_channels = in_channels
        self.net = nn.Sequential(nn.Conv2d(in_channels, 64, 7, 2, 3), nn.ReLU(),
                                 nn.MaxPool2d(3, 2, 1),
                                 nn.Conv2d(64, 64, 1, 1, 0), nn.ReLU(),
                                 nn.Conv2d(64, 192, 3, 1, 1), nn.ReLU(),
                                 nn.MaxPool2d(3, 2, 1),
                                 
                                 InceptionBlock(192, 96, 16, 64, 128, 32, 32),
                                 InceptionBlock(256, 128, 32, 128, 192, 96, 64),
                                 nn.MaxPool2d(3, 2, 1),
                                 InceptionBlock(480, 96, 16, 192, 208, 48, 64),
                                 InceptionBlock(512, 112, 24, 160, 224, 64, 64),
                                 InceptionBlock(512, 128, 24, 128, 256, 64, 64),
                                 InceptionBlock(512, 144, 32, 112, 288, 64, 64),
                                 
                                 nn.AvgPool2d(5, 3, 2), 
                                 nn.Conv2d(528, 128, 1),
                                 nn.Flatten(),
                                 nn.Linear(8192, 1024), nn.ReLU(),
                                 nn.Linear(1024, n_joints * 3))
        
    def forward(self, x):
        y = self.net(x)
        b, _ = y.shape
        y = y.reshape(b, 3, self.n_joints)
        return y
        

if __name__ == '__main__':
    net = GooglePoseNet(3, 24)
    net(torch.rand((32, 3, 360, 360)))