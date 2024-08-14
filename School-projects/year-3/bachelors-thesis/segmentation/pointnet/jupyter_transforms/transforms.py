import dgl.backend
import torch
import torch as t
import numpy as np
import dgl.geometry.fps as fps


class ToTensor(object):
    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        if t.cuda.is_available():
            return {"point_cloud": t.from_numpy(data).to(device='cuda', dtype=t.float),
                    "annotations": t.from_numpy(ground_truth).to(device='cuda', dtype=t.long)}
        return {"point_cloud": t.from_numpy(data).to(dtype=t.float),
                "annotations": t.from_numpy(ground_truth).to(dtype=t.long)}


class FPS(object):
    def __init__(self, points):
        self.points = points

    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        if data.size(0) == self.points:
            return {"point_cloud": data,
                    "annotations": ground_truth}

        idx = fps.farthest_point_sampler(data.unsqueeze(0).to('cpu'), self.points).squeeze()
        return {"point_cloud": data[idx].to(data.device),
                "annotations": ground_truth[idx]}


class Detach(object):
    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        data.requires_grad_(False),
        ground_truth.requires_grad_(False)
        return {"point_cloud": data,
                "annotations": ground_truth}


class Attach(object):
    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        data.requires_grad_(True),
        #ground_truth.requires_grad_(True)
        return {"point_cloud": data,
                "annotations": ground_truth}


class RandomHorizontalRotation(object):
    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        degree = (t.rand(1) * t.pi * 2)
        rotation_matrix = t.tensor([[t.cos(degree), 0, -t.sin(degree)],
                                    [0, 1, 0],
                                    [t.sin(degree), 0, t.cos(degree)]], device=data.device)
        data = t.matmul(data, rotation_matrix)
        return {"point_cloud": data, "annotations": ground_truth}


class Transpose(object):
    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        return {"point_cloud": t.permute(data, (1, 0)), "annotations": ground_truth}


class RandomCrop(object):
    def __init__(self, points=None):
        self.minimum_points_count = points

    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        points_count = data.size(0)
        minimum_points_count = points_count // 2 if self.minimum_points_count is None else self.minimum_points_count

        sample = self.offset_along_axis(0, minimum_points_count, sample)
        sample = self.offset_along_axis(1, minimum_points_count, sample)
        sample = self.offset_along_axis(2, minimum_points_count, sample)

        return sample

    @staticmethod
    def offset_along_axis(axis: int, minimum_points_count: int, sample: dict) -> dict:
        new_data, new_ground_truth = sample["point_cloud"].clone(), sample["annotations"].clone()
        offset = RandomOffset(axis)
        new_sample = offset({"point_cloud": new_data, "annotations": new_ground_truth})
        data = sample["point_cloud"]
        min_ = t.min(data, 0)[0]
        max_ = t.max(data, 0)[0]
        crop = Crop(min_, max_)
        new_sample = crop(new_sample)
        new_points_count = new_sample["point_cloud"].size(0)
        if new_points_count >= minimum_points_count:
            return new_sample
        return sample


class RandomOffset(object):
    def __init__(self, axis: int):
        self.axis = axis

    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        min_ = t.min(data, 0)[0]
        max_ = t.max(data, 0)[0]
        diff = max_ - min_
        offset = diff * (1.8 * t.rand(data.size(-1), device=data.device) - 0.9)

        data[:, self.axis] += offset[self.axis]
        return {"point_cloud": data, "annotations": ground_truth}


class Offset(object):
    def __init__(self, vector: t.tensor):
        self.vector = vector

    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        return {"point_cloud": data + self.vector, "annotations": ground_truth}


class Crop(object):
    def __init__(self, min_=-1, max_=1):
        self.min_ = min_
        self.max_ = max_

    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        mask = t.logical_and(data >= self.min_, data <= self.max_)
        mask = t.sum(mask, dim=1) == 3
        indexes = t.nonzero(mask).squeeze()
        cropped_data, cropped_ground_truth = data[indexes], ground_truth[indexes]
        return {"point_cloud": cropped_data, "annotations": cropped_ground_truth}


class Center(object):
    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        center = t.sum(data, 0) / data.size(0)
        data -= center
        return {"point_cloud": data, "annotations": ground_truth}


class Normalise(object):
    def __init__(self, min_: t.tensor, max_: t.tensor):
        self.min_ = min_
        self.max_ = max_

    def __call__(self, sample: dict) -> dict:
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        return {"point_cloud": (2 * data - (self.max_ + self.min_)) / (self.max_ - self.min_), "annotations": ground_truth}


class ToArray(object):
    def __call__(self, sample: dict) -> (np.array, np.array):
        data, ground_truth = sample["point_cloud"], sample["annotations"]
        return data.detach().numpy(), ground_truth.numpy()

