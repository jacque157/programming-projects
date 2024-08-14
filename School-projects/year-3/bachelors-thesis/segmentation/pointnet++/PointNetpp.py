from PointNet2Modules import *
import torch
import torch.nn as nn


class PartSegmentation(nn.Module):
    def __init__(self, input_dimension=3, labels=15):
        super(PartSegmentation, self).__init__()

        self.sa1 = SetAbstraction(512, 0.2, 32, (input_dimension + input_dimension, 64, 64, 128))
        self.sa2 = SetAbstraction(128, 0.4, 64, (128 + 3, 128, 128, 256))
        self.sa3 = GlobalSetAbstraction((256 + 3, 256, 512, 1024))

        self.fp1 = FeaturePropagation((1024 + 256, 256, 256))
        self.fp2 = FeaturePropagation((256 + 128, 256, 128))
        self.fp3 = FeaturePropagation((128 + 3 + 3, 128, 128, 128))

        self.pointnet = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, labels, 1),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, point_cloud):
        points1, features1 = self.sa1(torch.cat((point_cloud, point_cloud), dim=1))
        points2, features2 = self.sa2(torch.cat((points1, features1), dim=1))
        points3, features3 = self.sa3(torch.cat((points2, features2), dim=1))

        features4 = self.fp1(points2, points3, features2, features3)
        features5 = self.fp2(points1, points2, features1, features4)
        features6 = self.fp3(point_cloud, points1, torch.cat((point_cloud, point_cloud), 1), features5)

        features = self.pointnet(features6)

        return features, features3


def main():
    pc = torch.rand(32, 3, 2048)
    net = PartSegmentation()
    res1, res2 = net(pc)
    print(res1.permute(0, 2, 1).shape)
    print(res2.permute(0, 2, 1).shape)


if __name__ == '__main__':
    main()