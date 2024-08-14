import torch
import torch.nn as  nn
import torch.nn.functional as F


class GoogleNetBase(nn.Module):
    def __init__(self, n_channels, n_joints, pretrained=True):
        super().__init__()
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels
        self.network = None
        self.regression = None

    def forward(self, x):
        x = self.network.conv1(x)
        x = self.network.maxpool1(x)
        x = self.network.conv2(x)
        x = self.network.conv3(x)
        x = self.network.maxpool2(x)

        x = self.network.inception3a(x)
        x = self.network.inception3b(x)
        x = self.network.maxpool3(x)
        
        x = self.network.inception4a(x)
        x = self.network.inception4b(x)
        x = self.network.inception4c(x)
        x = self.network.inception4d(x)
        x = self.network.inception4e(x)
        x = self.network.maxpool4(x)

        x = self.network.inception5a(x)
        x = self.network.inception5b(x)
        x = self.network.avgpool(x) 

        x = torch.flatten(x, 1)
        b, _ = x.shape
        x = self.regression(x)

        return torch.reshape(x, (b, 3, self.n_joints))

    def loss(self, predicted_skeletons, gt_skeletons):
        return F.mse_loss(predicted_skeletons, gt_skeletons)

class GoogleNet(GoogleNetBase):
    def __init__(self, n_channels, n_joints, pretrained=True):
        super().__init__(n_channels, n_joints, pretrained)
        self.network = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=pretrained)
        if n_channels != 3:
            self.network.conv1.conv = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.network.avgpool = nn.AvgPool2d(5, 3, 2)
        self.regression = nn.Sequential(nn.Linear(6*6*1024, 1024),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(),

                                        nn.Linear(1024, n_joints * 3))
        
        del self.network.dropout
        del self.network.fc
        
if __name__ == '__main__':
    channels = 6
    img = torch.rand(16, channels, 512, 512)
    model = GoogleNet(channels, 22)
    out = model(img)
