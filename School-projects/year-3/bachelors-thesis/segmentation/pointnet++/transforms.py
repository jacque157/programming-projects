import torch
import numpy as np


class ToTensor(object):
    def __call__(self, np_array: np.array):
        return torch.from_numpy(np_array)

