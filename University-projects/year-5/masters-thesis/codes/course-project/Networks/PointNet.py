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

class PointNetPoseRegression(nn.Module):
    def __init__(self, n_channels, n_joints, device) -> None:
        super(PointNetPoseRegression, self).__init__()
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
        self.fc2 = FCL(512, 256, dropout=False)
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

    def loss(self, predictions, ground_truth, matrix):
        regression_loss = torch.nn.functional.mse_loss(predictions, ground_truth)
        weight = 0.001
        device = predictions.device
        identity = torch.eye(matrix.size(1)).to(device)
        reg_term = torch.norm(identity - torch.bmm(matrix, matrix.transpose(2, 1)))
        return regression_loss + reg_term * weight

        
