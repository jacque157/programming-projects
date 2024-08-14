from torch import nn
import torch


class PointNetLoss(nn.Module):
    def __init__(self, device, weight=0.001):
        super(PointNetLoss, self).__init__()
        self.weight = weight
        self.nl_loss = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, trans_matrix, prediction, ground_truth):
        identity = torch.eye(trans_matrix.size(1)).to(self.device)
        reg_term = torch.norm(identity - torch.bmm(trans_matrix, trans_matrix.transpose(2, 1)))
        loss = self.nl_loss(prediction, ground_truth) + reg_term * self.weight
        return loss
