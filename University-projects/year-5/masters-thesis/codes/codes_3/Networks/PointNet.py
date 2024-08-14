import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, batch_normalisation=True, dropout=False):
        super(MLP, self).__init__()
        self.batch_norm = nn.BatchNorm1d(output_size) if batch_normalisation else None
        self.convolution = nn.Conv1d(input_size, output_size, (1,))
        self.dropout = nn.Dropout(0.3) if dropout else None
        self.relu = torch.nn.ReLU()

    def forward(self, batch):
        batch = self.convolution(batch)
        if self.batch_norm:
            if self.dropout:
                batch = self.dropout(batch)
            return self.relu(self.batch_norm(batch))
        return batch


class FCL(nn.Module):
    def __init__(self, input_size, output_size, batch_normalisation=True, dropout=False):
        super(FCL, self).__init__()
        self.batch_norm = nn.BatchNorm1d(output_size) if batch_normalisation else None
        self.dropout = nn.Dropout(0.3) if dropout else None
        self.perceptron = nn.Linear(input_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, batch):
        batch = self.perceptron(batch)
        if self.batch_norm:
            if self.dropout:
                batch = self.dropout(batch)
            return self.relu(self.batch_norm(batch))
        return batch


class TNet(nn.Module):
    def __init__(self, device, dimensions_length=3):
        super(TNet, self).__init__()
        self.device = device
        self.dimension = dimensions_length

        self.mlp1 = MLP(dimensions_length, 64)
        self.mlp2 = MLP(64, 128)
        self.mlp3 = MLP(128, 1024)

        self.fc1 = FCL(1024, 512)
        self.fc2 = FCL(512, 256, dropout=True)
        self.fc3 = FCL(256, self.dimension * self.dimension, batch_normalisation=False)

    def forward(self, batch):
        a = self.mlp1(batch)
        a = self.mlp2(a)
        a = self.mlp3(a)
        a = torch.max(a, 2)[0]
        a = self.fc1(a)
        a = self.fc2(a)
        a = self.fc3(a)

        batch_size = batch.shape[0]
        identity_batch = torch.eye(self.dimension, self.dimension).repeat(batch_size, 1, 1)
        if a.is_cuda:
            identity_batch = identity_batch.to(self.device)

        a = a.view(-1, self.dimension, self.dimension)
        a += identity_batch
        return a


class PointNetSegmentation(nn.Module):
    def __init__(self, device, scores):
        super(PointNetSegmentation, self).__init__()
        self.device = device

        self.tnet3 = TNet(device, 3)
        self.mlp1 = nn.Sequential(
            MLP(3, 64, dropout=True),
            MLP(64, 64)
        )
        self.tnet64 = TNet(device, 64)
        self.mlp2 = nn.Sequential(
            MLP(64, 64),
            MLP(64, 128),
            MLP(128, 1024)
        )
        self.mlp3 = nn.Sequential(
            MLP(1088, 512),
            MLP(512, 256),
            MLP(256, 128, dropout=True)
        )
        self.mlp4 = nn.Sequential(
            MLP(128, 128, dropout=True),
            MLP(128, scores)
        )

    def forward(self, batch):
        batch_c = batch.clone()
        batch_mat3x3 = self.tnet3(batch_c)
        x = torch.matmul(batch_mat3x3, batch)
        x = self.mlp1(x)
        x_c = x.clone()
        batch_mat64x64 = self.tnet64(x_c)
        x = torch.matmul(batch_mat64x64, x)
        features = x.clone()
        x = self.mlp2(x)
        global_features = torch.max(x, 2, True)[0]
        global_features = global_features.repeat(1, 1, features.size(2))
        local_features = torch.cat((features, global_features), 1)
        x = self.mlp3(local_features)
        x = self.mlp4(x)

        return batch_mat3x3, batch_mat64x64, x


class PointNetSegmentationVer1(nn.Module):
    def __init__(self, device, scores):
        super(PointNetSegmentationVer1, self).__init__()
        self.device = device

        self.tnet3 = TNet(device, 3)
        self.mlp1 = nn.Sequential(
            MLP(3, 64),
            MLP(64, 64),
            MLP(64, 128)
        )
        self.tnet128 = TNet(device, 128)
        self.mlp2 = nn.Sequential(
            MLP(128, 128),
            MLP(128, 256),
            MLP(256, 1024)
        )
        self.mlp3 = nn.Sequential(
            MLP(1152, 1024),
            MLP(1024, 512),
            MLP(512, 256)
        )
        self.mlp4 = nn.Sequential(
            MLP(256, 128, dropout=True),
            MLP(128, 128, dropout=True),
            MLP(128, scores)
        )

    def forward(self, batch):
        batch_c = batch.clone()
        batch_mat3x3 = self.tnet3(batch_c)
        x = torch.matmul(batch_mat3x3, batch)
        x = self.mlp1(x)
        x_c = x.clone()
        batch_mat128x128 = self.tnet128(x_c)
        x = torch.matmul(batch_mat128x128, x)
        features = x.clone()
        x = self.mlp2(x)
        global_features = torch.max(x, 2, True)[0]
        global_features = global_features.repeat(1, 1, features.size(2))
        local_features = torch.cat((features, global_features), 1)
        x = self.mlp3(local_features)
        x = self.mlp4(x)

        return batch_mat3x3, batch_mat128x128, x


class PointNetPartSegmentation(nn.Module):
    def __init__(self, device, scores):
        super(PointNetPartSegmentation, self).__init__()
        self.device = device

        self.tnet3 = TNet(device, 3)
        self.mlp1 = MLP(3, 64)
        self.mlp2 = MLP(64, 128)
        self.mlp3 = MLP(128, 128)
        self.tnet128 = TNet(device, 128)
        self.mlp4 = MLP(128, 512)
        self.mlp5 = MLP(512, 2048)

        self.mlp6 = nn.Sequential(
            MLP(3008, 256),
            MLP(256, 256, dropout=True),
            MLP(256, 128, dropout=True),
            MLP(128, scores)
        )

    def forward(self, batch):
        batch_c = batch.clone()
        batch_mat3x3 = self.tnet3(batch_c)
        x = torch.matmul(batch_mat3x3, batch)
        x = self.mlp1(x)
        features_64 = x.clone()
        x = self.mlp2(x)
        features_128_1 = x.clone()
        x = self.mlp3(x)
        features_128_2 = x.clone()
        x_c = x.clone()
        batch_mat128x128 = self.tnet128(x_c)
        x = torch.matmul(batch_mat128x128, x)
        features_128_3 = x.clone()
        x = self.mlp4(x)
        features_512 = x.clone()
        x = self.mlp5(x)
        global_features = torch.max(x, 2, True)[0]
        global_features = global_features.repeat(1, 1, features_64.size(2))
        local_features = torch.cat((features_64, features_128_1, features_128_2,
                                    features_128_3, features_512, global_features), 1)
        x = self.mlp6(local_features)

        return batch_mat3x3, batch_mat128x128, x

class PointNetPartSegmentationReduced(nn.Module):
    def __init__(self, device, scores):
        super(PointNetPartSegmentationReduced, self).__init__()
        self.device = device

        self.tnet3 = TNet(device, 3)
        self.mlp1 = MLP(3, 32)
        self.mlp2 = MLP(32, 64)
        self.tnet64 = TNet(device, 64)
        self.mlp3 = MLP(64, 256)
        self.mlp4 = MLP(256, 1024)

        self.mlp5 = nn.Sequential(
            MLP(1440, 256),
            MLP(256, 128, dropout=True),
            MLP(128, 128, dropout=True),
            MLP(128, scores)
        )

    def forward(self, batch):
        batch_c = batch.clone()
        batch_mat3x3 = self.tnet3(batch_c)
        x = torch.matmul(batch_mat3x3, batch)
        x = self.mlp1(x)
        features_32 = x.clone()
        x = self.mlp2(x)
        features_64_1 = x.clone()
        x_c = x.clone()
        batch_mat64x64 = self.tnet64(x_c)
        x = torch.matmul(batch_mat64x64, x)
        features_64_2 = x.clone()
        x = self.mlp3(x)
        features_256 = x.clone()
        x = self.mlp4(x)
        global_features = torch.max(x, 2, True)[0]
        global_features = global_features.repeat(1, 1, features_32.size(2))
        local_features = torch.cat((features_32, features_64_1, features_64_2,
                                    features_256, global_features), 1)
        x = self.mlp5(local_features)

        return batch_mat3x3, batch_mat64x64, x


class PointPoseNet(nn.Module):
    def __init__(self, n_channels, n_joints, device) -> None:
        super(PointPoseNet, self).__init__()
        self.n_joints = self.joints = n_joints
        self.n_channels = n_channels 
        self.device = device

        self.tnet3 = TNet(device, n_channels)
        self.mlp1 = nn.Sequential(
            MLP(n_channels, 64, dropout=True),
            MLP(64, 64)
        )
        self.tnet64 = TNet(device, 64)
        self.mlp2 = nn.Sequential(
            MLP(64, 64),
            MLP(64, 128),
            MLP(128, 1024)
        )
     
        self.fc1 = FCL(1024, 512)
        self.fc2 = FCL(512, 256, dropout=True)
        self.fc3 = FCL(256, n_joints * 3, batch_normalisation=False)


    def forward(self, batch):
        batch_c = batch.clone()
        batch_mat3x3 = self.tnet3(batch_c)
        x = torch.matmul(batch_mat3x3, batch)
        x = self.mlp1(x)
        x_c = x.clone()
        batch_mat64x64 = self.tnet64(x_c)
        x = torch.matmul(batch_mat64x64, x)
        x = self.mlp2(x)
        global_features = torch.max(x, 2, True)[0].flatten(1)
        x = self.fc1(global_features)
        x = self.fc2(x)
        x = self.fc3(x)
        b, _ = x.shape
        poses = torch.reshape(x, (b, 3, self.joints))
        return batch_mat3x3, batch_mat64x64, poses

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.rand((10, 3, 1000))
    net = TNet(device, 3)
    y = net(x)
    print("T-Net 3", y.shape)

    x = torch.rand((10, 64, 1000))
    net = TNet(device, 64)
    y = net(x)
    print("T-Net 64", y.shape)

    x = torch.rand((10, 3, 1000))
    net = PointNetSegmentation(device, 15)
    m3, m128, y = net(x)
    print("PointNet Class", y.shape)
    print("mat 3", m3.shape)
    print("mat 128", m128.shape)

    x = torch.rand((10, 3, 1000))
    net = PointNetPartSegmentation(device, 15)
    m3, m128, y = net(x)
    print("PointNet Class", y.shape)
    print("mat 3", m3.shape)
    print("mat 128", m128.shape)

    x = torch.rand((10, 3, 1000))
    net = PointNetPartSegmentationReduced(device, 15)
    m3, m64, y = net(x)
    print("PointNet Class", y.shape)
    print("mat 3", m3.shape)
    print("mat 64", m64.shape)

if __name__ == '__main__':
    main()
    """optim = torch.optim.Adam()
    optim.zero_grad()
    torch.optim.NAdam()"""











