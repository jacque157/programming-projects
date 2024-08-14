import torch
from torch import nn

from ResNet import ResNet50



class MeshNet(nn.Module):
    def __init__(self, joints, in_channels=3) -> None:
        super().__init__()
        