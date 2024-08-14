import torch
import torch.nn as nn
from torch.autograd import Function
import funcs


class SetAbstraction(nn.Module):
    def __init__(self, regions, radius, samples_count, mini_pointnet_dimensions):
        super(SetAbstraction, self).__init__()
        self.r = radius
        self.k = regions
        self.n = samples_count
        self.pointnet = MiniPointNet(mini_pointnet_dimensions, False)

    def forward(self, point_cloud):
        point_cloud = point_cloud.permute(0, 2, 1)
        centroids, sampled_cloud = funcs.sample_and_group(point_cloud, self.k, self.r, self.n)
        sampled_cloud = sampled_cloud.permute(0, 3, 2, 1)
        features = self.pointnet(sampled_cloud)
        return centroids.permute(0, 2, 1), features


class MultiScaleGrouping:
    pass


class GlobalSetAbstraction(nn.Module):
    def __init__(self, mini_pointnet_dimensions):
        super(GlobalSetAbstraction, self).__init__()
        self.pointnet = MiniPointNet(mini_pointnet_dimensions, False)

    def forward(self, point_cloud):
        point_cloud = point_cloud.permute(0, 2, 1)
        centroids, sampled_cloud = funcs.sample_and_group_all(point_cloud)
        sampled_cloud = sampled_cloud.permute(0, 3, 2, 1)
        features = self.pointnet(sampled_cloud)
        return centroids.permute(0, 2, 1), features


class FullyConnectedLayer:
    pass


class FeaturePropagation(nn.Module):
    def __init__(self, mini_pointnet_dimensions):
        super(FeaturePropagation, self).__init__()
        self.pointnet = MiniPointNet(mini_pointnet_dimensions)

    def forward(self, centroids1, centroids2, features1, features2):
        device = centroids1.device
        centroids1 = centroids1.permute(0, 2, 1)
        centroids2 = centroids2.permute(0, 2, 1)
        features2 = features2.permute(0, 2, 1)

        batch_size, points_count1, dimensions = centroids1.shape
        _, points_count2, _ = centroids2.shape

        if points_count2 == 1:
            interpolation = features2.repeat(1, points_count1, 1)
        else:
            distances = funcs.square_distance(centroids1, centroids2)
            distances, indexes = distances.sort(dim=-1)
            distances, indexes = distances[:, :, :3], indexes[:, :, :3]

            distances_reciprocal = 1 / (distances + 1e-8)
            weight = distances_reciprocal / torch.sum(distances_reciprocal, dim=2, keepdim=True)

            batch_indexes = torch.arange(batch_size, device=device).repeat_interleave(points_count1 * 3)

            interpolation = torch.sum(features2[batch_indexes, indexes.flatten()].reshape(
                batch_size, points_count1, 3, features2.size(-1)) * weight.view(batch_size, points_count1, 3, 1), dim=2)

        if features1 is not None:
            features1 = features1.permute(0, 2, 1)
            features = torch.cat((features1, interpolation), dim=-1)
        else:
            features = interpolation

        features = features.permute(0, 2, 1)
        features = self.pointnet(features, False)

        return features


class MiniPointNet(nn.Module):
    def __init__(self, mini_pointnet_dimensions, one_dimensional_kernel=True):
        super(MiniPointNet, self).__init__()
        in_channels = None
        self.pointnet = nn.Sequential()
        for i, dimension in enumerate(mini_pointnet_dimensions):
            if in_channels is None:
                in_channels = dimension
            else:
                out_channels = dimension
                if one_dimensional_kernel:
                    self.pointnet.add_module(f"MLP({in_channels}, {out_channels})",
                                             nn.Conv1d(in_channels, out_channels, 1))
                    self.pointnet.add_module(f"BatchNorm{i}", nn.BatchNorm1d(out_channels))
                else:
                    self.pointnet.add_module(f"MLP({in_channels}, {out_channels})",
                                             nn.Conv2d(in_channels, out_channels, 1))
                    self.pointnet.add_module(f"BatchNorm{i}", nn.BatchNorm2d(out_channels))
                self.pointnet.add_module(f"ReLU{i}", nn.ReLU())
                in_channels = out_channels

    def forward(self, point_clouds, max_pool=True):
        features = self.pointnet(point_clouds)
        if max_pool:
            global_features = torch.max(features, 2)[0]
            return global_features
        return features


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.nl_loss = nn.CrossEntropyLoss()

    def forward(self, prediction, ground_truth):
        return self.nl_loss(prediction, ground_truth)




