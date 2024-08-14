import numpy as np
import torch


class AddMask:
    def __call__(self, data):
        if 'point_clouds' in data:
            pt_clouds = data['point_clouds']
            assert len(pt_clouds.shape) == 4
            xs = pt_clouds[:, :, :, 0]
            ys = pt_clouds[:, :, :, 1]
            zs = pt_clouds[:, :, :, 2]

            mask = np.logical_not(np.logical_and(xs == 0.0, np.logical_and(ys == 0.0, zs == 0.0)))
            data['point_clouds'] = np.concatenate((pt_clouds, mask[:, :, :, None]), 3)
        else:
            images = data['images']
            assert len(images.shape) == 4
            xs = images[:, :, :, 0]
            ys = images[:, :, :, 1]
            zs = images[:, :, :, 2]

            mask = np.logical_not(np.logical_and(xs == 0.0, np.logical_and(ys == 0.0, zs == 0.0)))
            data['images'] = np.concatenate((images, mask[:, :, :, None]), 3)
        return data

class AddSegmentations:
    def __call__(self, data):
        if 'point_clouds' in data:
            pt_clouds = data['point_clouds']
            assert len(pt_clouds.shape) == 4
            segmentations = data['segmentations']
            
            data['point_clouds'] = np.concatenate((pt_clouds, segmentations), 3)
        else:
            images = data['images']
            assert len(images.shape) == 4
            segmentations = data['segmentations']
            
            data['images'] = np.concatenate((images, segmentations), 3)
        return data

class ZeroCenter:
    def __init__(self, mean, transform_skeletons=False):
        self.mean = mean
        self.transform_skeletons = transform_skeletons

    def __call__(self, data):
        if 'point_clouds' in data:
            n = len(data['point_clouds'].shape)
            if n == 3:
                data['point_clouds'][:, :, 0:3] -= self.mean
            elif n == 4:
                data['point_clouds'][:, :, :, 0:3] -= self.mean
            elif n == 5:
                data['point_clouds'][:, :, :, :, 0:3] -= self.mean
        if 'images' in data:
            n = len(data['images'].shape)
            if n == 3:
                data['images'][:, :, 0:3] -= self.mean
            elif n == 4:
                data['images'][:, :, :, 0:3] -= self.mean
            elif n == 5:
                data['images'][:, :, :, :, 0:3] -= self.mean
        if self.transform_skeletons:
            data['skeletons'][:, :, 0:3] -= self.mean
        data['center'] = self.mean
        return data

class Rescale:
    def __init__(self, min_, max_, a=-1, b=1, transform_skeletons=False):
        self.a = np.array(a)
        self.b = np.array(b)
        self.max_ = max_
        self.min_ = min_
        self.transform_skeletons = transform_skeletons

    def __call__(self, data):
        factor = (self.b - self.a) / (self.max_ - self.min_)
        if 'point_clouds' in data:
            n = len(data['point_clouds'].shape)
            if n == 3:
                data['point_clouds'][:, :, 0:3] = self.a + ((data['point_clouds'][:, :, 0:3] - self.min_) * factor)
            elif n == 4:
                data['point_clouds'][:, :, :, 0:3] = self.a + ((data['point_clouds'][:, :, :, 0:3] - self.min_) * factor)
            elif n == 5:
                data['point_clouds'][:, :, :, :, 0:3] = self.a + ((data['point_clouds'][:, :, :, :, 0:3] - self.min_) * factor)
        if 'images' in data:
            n = len(data['images'].shape)
            if n == 3:
                data['images'][:, :, 0:3] = self.a + ((data['images'][:, :, 0:3] - self.min_) * factor)
            elif n == 4:
                data['images'][:, :, :, 0:3] = self.a + ((data['images'][:, :, :, 0:3] - self.min_) * factor)
            elif n == 5:
                data['images'][:, :, :, :, 0:3] = self.a + ((data['images'][:, :, :, :, 0:3] - self.min_) * factor)
        if self.transform_skeletons:
            data['skeletons'][:, :, 0:3] = self.a + ((data['skeletons'][:, :, 0:3] - self.min_) * factor)
        data['a'] = self.a
        data['b'] = self.b
        data['min_'] = self.min_
        data['max_'] = self.max_
        return data

class Standardization:
    def __init__(self, mean, std, transform_skeletons=False):
        self.mean = mean
        self.std = std
        self.transform_skeletons = transform_skeletons

    def __call__(self, data):
        if 'point_clouds' in data:
            n = len(data['point_clouds'].shape)
            if n == 3:
                data['point_clouds'][:, :, 0:3] = (data['point_clouds'][:, :, 0:3] - self.mean) / self.std
            elif n == 4:
                data['point_clouds'][:, :, :, 0:3] = (data['point_clouds'][:, :, :, 0:3] - self.mean) / self.std
            elif n == 5:
                data['point_clouds'][:, :, :, :, 0:3] = (data['point_clouds'][:, :, :, :, 0:3] - self.mean) / self.std
        if 'images' in data:
            n = len(data['images'].shape)
            if n == 3:
                data['images'][:, :, 0:3] = (data['images'][:, :, 0:3] - self.mean) / self.std
            elif n == 4:
                data['images'][:, :, :, 0:3] = (data['images'][:, :, :, 0:3] - self.mean) / self.std
            elif n == 5:
                data['images'][:, :, :, :, 0:3] = (data['images'][:, :, :, :, 0:3] - self.mean) / self.std
        if self.transform_skeletons:
            data['skeletons'][:, :, 0:3] = (data['skeletons'][:, :, 0:3] - self.mean) / self.std
        data['center'] = self.mean
        data['std'] = self.std
        return data
        
class ToTensor:      
    def __call__(self, data):
        for key, value in data.items():
            if key == 'skeleton_depths':
                data[key] = data[key][:, :, None]
            if key in ('point_clouds', 'images', 'skeletons',
                       'heat_maps', 'skeleton_depths',
                       '2D_skeletons_depths', '2D_skeletons_offsets'):
                if len(data[key].shape) == 5:
                    data[key] = np.moveaxis(data[key], -1, 2)
                else:
                    data[key] = np.moveaxis(data[key], -1, 1)

            if key in ("sequences", "frames"):
                data[key] = torch.from_numpy(data[key])
                data[key] = data[key].long()
            elif key == 'name':
                data[key] = data[key]
            else:
                data[key] = torch.from_numpy(data[key])
                data[key] = data[key].float()
        return data

class ToNumpy:
    def __init__(self, force=True):
        self.force = force

    def __call__(self, data):
        for key, value in data.items():
            if key == 'name':
                continue
            data[key] = data[key].numpy(force=self.force)
            if len(data[key].shape) >= 3:
                if key in ('point_clouds', 'images', 'skeletons',
                       'heat_maps', 'skeleton_depths',
                       '2D_skeletons_depths', '2D_skeletons_offsets'):
                    if len(data[key].shape) == 5:
                        data[key] = np.moveaxis(data[key], -1, 2).squeeze()
                    else:
                        data[key] = np.moveaxis(data[key], -1, 1).squeeze()
        return data

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data

class RemoveHandJoints:
    def __call__(self, data):
        data['skeletons'] = data['skeletons'][:, 0 : 22, :]
        if 'heat_maps' in data:
            if len(data['heat_maps'].shape) == 5:
                data['heat_maps'] = data['heat_maps'][:, :, :, :, 0 : 22]
            else:
                data['heat_maps'] = data['heat_maps'][:, :, :, 0 : 22]
        if 'skeletons_2D' in data:
            if len(data['skeletons_2D'].shape) == 5:
                data['skeletons_2D'] = data['skeletons_2D'][:, :, :, :, 0 : 22]
            else:
                data['skeletons_2D'] = data['skeletons_2D'][:, :, :, 0 : 22]
        if 'skeletons_depths' in data:
            if len(data['skeletons_depths'].shape) == 5:
                data['skeletons_depths'] = data['skeletons_depths'][:, :, :, :, 0 : 22]
            else:
                data['skeletons_depths'] = data['skeletons_depths'][:, :, :, 0 : 22]
        return data
    
class AddNormalNoise:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        if 'point_clouds' in data:
            point_cloud = data['point_clouds']
            noise = ((2 * np.random.randn(*point_cloud.shape)) - 1) * self.sigma
            data['point_clouds'] = point_cloud + noise
        if 'images' in data:
            images = data['images']
            noise = ((2 * np.random.randn(*images.shape)) - 1) * self.sigma
            data['images'] = images + noise
        return data

class ZeroPad:
    def __init__(self, output_shape, input_shape):    
        self.padding = None
        self.output_shape = output_shape
        if input_shape is not None:
            h_i, w_i = input_shape
            h_o, w_o = output_shape
            h_pl, w_pl = (h_o - h_i) // 2, (w_o - w_i) // 2
            h_pr, w_pr = h_o - h_i - h_pl, w_o - w_i - w_pl
            self.padding = (0, 0), (h_pl, h_pr), (w_pl, w_pr), (0, 0)

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:  
        padding = self.padding
        if padding is None:
            if 'point_clouds' in sample:
                s, h_i, w_i, c = sample['point_clouds'].shape
            else:
                s, h_i, w_i, c = sample['images'].shape
            h_o, w_o = self.output_shape
            h_pl, w_pl = (h_o - h_i) // 2, (w_o - w_i) // 2
            h_pr, w_pr = h_o - h_i - h_pl, w_o - w_i - w_pl
            padding = (0, 0), (h_pl, h_pr), (w_pl, w_pr), (0, 0)

        if 'point_clouds' in sample:
            sample['point_clouds'] = np.pad(sample['sequences'], padding)
        if 'images' in sample:
            sample['images'] = np.pad(sample['images'], padding)
        return sample
