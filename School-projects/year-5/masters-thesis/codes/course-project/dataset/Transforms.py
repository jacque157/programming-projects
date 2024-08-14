import numpy as np
import torch


class AddMask:
    def __call__(self, data):
        pt_clouds = data['point_clouds']
        assert len(pt_clouds.shape) == 4
        xs = pt_clouds[:, :, :, 0]
        ys = pt_clouds[:, :, :, 1]
        zs = pt_clouds[:, :, :, 2]

        mask = np.logical_not(np.logical_and(xs == 0.0, np.logical_and(ys == 0.0, zs == 0.0)))
        data['point_clouds'] = np.concatenate((pt_clouds, mask[:, :, :, None]), 3)
        return data

class AddSegmentations:
    def __call__(self, data):
        pt_clouds = data['point_clouds']
        assert len(pt_clouds.shape) == 4
        segmentations = data['segmentations']
        
        data['point_clouds'] = np.concatenate((pt_clouds, segmentations), 3)
        return data

class ZeroCenter:
    def __init__(self, mean, transform_skeletons=False):
        self.mean = mean
        self.transform_skeletons = transform_skeletons

    def __call__(self, data):
        data['point_clouds'][:, :, 0:3] -= self.mean
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
        data['point_clouds'][:, :, 0:3] = self.a + ((data['point_clouds'][:, :, 0:3] - self.min_) * factor)
        if self.transform_skeletons:
            data['skeletons'][:, :, 0:3] = self.a + ((data['skeletons'][:, :, 0:3] - self.min_) * factor)
        data['a'] = self.a
        data['b'] = self.b
        data['min_'] = self.min_
        data['max_'] = self.max_
        return data

class ToTensor:      
    def __call__(self, data):
        for key, value in data.items():
            if key == 'skeleton_depths':
                data[key] = data[key][:, :, None]
            if len(data[key].shape) >= 3 and key != 'rotation_matrix_inverted':
                #print()
                #print(key)
                #print('to tensor')
                #print(data[key].shape)
                data[key] = np.moveaxis(data[key], -1, 1)
                #print(data[key].shape)
              
            #data[key] = torch.from_numpy(data[key])                
            if key in ("sequences", "frames"):
                data[key] = torch.from_numpy(data[key])
                data[key] = data[key].long()
            elif key == 'name':
                data[key] = data[key]
                """elif key == 'rotation_matrix_inverted':
                R_inv = data[key]
                n, _, _ = R_inv.shape        
                data[key] = torch.from_numpy(np.copy(R_inv[0])).expand(n, 3, 3).float()"""
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
                #print()
                #print('to numpy')
                #print(data[key].shape)
                data[key] = np.moveaxis(data[key], 1, -1).squeeze()
                #print(data[key].shape)
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
            data['heat_maps'] = data['heat_maps'][:, :, :, 0 : 22]
        if 'skeletons_2D' in data:
            data['skeletons_2D'] = data['skeletons_2D'][:, 0 : 22]
        if 'skeletons_depths' in data:
            data['skeletons_depths'] = data['skeletons_depths'][:, 0 : 22]
        return data
    
class RandomSampling:
    def __init__(self, n):
        self.n = n

    def __call__(self, data):
        point_cloud = data['point_clouds']
        assert len(point_cloud.shape) == 3
        point_cloud = np.swapaxes(point_cloud, 0, 1)
        np.random.shuffle(point_cloud)
        point_cloud = np.swapaxes(point_cloud, 1, 0)
        data['point_clouds'] = point_cloud[:, 0 : self.n, :]
        return data

class RandomRotation:
    def __init__(self, max_alpha, max_beta, max_gamma=180):
        self.max_alpha = (max_alpha / 360) * 2 * np.pi
        self.max_beta = (max_beta / 360) * 2 * np.pi
        self.max_gamma = (max_gamma / 360) * 2 * np.pi

    def __call__(self, data): 
        point_cloud = data['point_clouds']
        s, n, c = point_cloud.shape
        assert  c == 3
        rotation = np.repeat(np.eye(3)[None], s, 0)
        if self.max_alpha:
            alpha = ((2 * np.random.rand(s)) - 1) * self.max_alpha
            cos_ = np.cos(alpha)
            sin_ = np.sin(alpha)
            rot_x = np.repeat(np.eye(3)[None], s, 0)
            rot_x[:, 1, 1] = cos_
            rot_x[:, 1, 2] = -sin_
            rot_x[:, 2, 1] = sin_
            rot_x[:, 2, 2] = cos_
            rotation = np.matmul(rotation, rot_x)
        if self.max_beta:
            beta = ((2 * np.random.rand(s)) - 1) * self.max_beta
            cos_ = np.cos(beta)
            sin_ = np.sin(beta)
            rot_y = np.repeat(np.eye(3)[None], s, 0)
            rot_y[:, 0, 0] = cos_
            rot_y[:, 0, 2] = sin_
            rot_y[:, 2, 0] = -sin_
            rot_y[:, 2, 2] = cos_
            rotation = np.matmul(rotation, rot_y)
        if self.max_gamma:
            gamma = ((2 * np.random.rand(s)) - 1) * self.max_gamma
            cos_ = np.cos(gamma)
            sin_ = np.sin(gamma)
            rot_z = np.repeat(np.eye(3)[None], s, 0)
            rot_z[:, 0, 0] = cos_
            rot_z[:, 0, 1] = -sin_
            rot_z[:, 1, 0] = sin_
            rot_z[:, 1, 1] = cos_
            rotation = np.matmul(rotation, rot_z)
        point_cloud = data['point_clouds']
        data['point_clouds'] = np.matmul(point_cloud, rotation) 
        skeletons = data['skeletons']
        data['skeletons'] = np.matmul(skeletons, rotation)
        return data

class AddNormalNoise:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, data):
        point_cloud = data['point_clouds']
        noise = ((2 * np.random.randn(*point_cloud.shape)) - 1) * self.sigma
        data['point_clouds'] = point_cloud + noise
        return data

class ZeroPad:
    def __init__(self, output_shape, input_shape):    
        """self.pad = None
        self.output_shape = output_shape
        if input_shape is not None:
            h_i, w_i = input_shape
            h_o, w_o = output_shape
            h_p, w_p = (h_o - h_i) // 2, (w_o - w_i) // 2
            self.pad = nn.ZeroPad2d((h_p, w_p))"""
        self.padding = None
        self.output_shape = output_shape
        if input_shape is not None:
            h_i, w_i = input_shape
            h_o, w_o = output_shape
            h_pl, w_pl = (h_o - h_i) // 2, (w_o - w_i) // 2
            h_pr, w_pr = h_o - h_i - h_pl, w_o - w_i - w_pl
            self.padding = (0, 0), (h_pl, h_pr), (w_pl, w_pr), (0, 0)

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:  
        """if self.pad is not None:
            pad = self.pad
        else:
            s, h_i, w_i, c = sample['sequences'].shape
            h_o, w_o = self.output_shape
            h_p, w_p = (h_o - h_i) // 2, (w_o - w_i) // 2
            pad = nn.ZeroPad2d((h_p, w_p))
        
        sample['sequences'] = pad(sample['sequences'])
        if 'segmentations' in sample:
            sample['segmentations'] = pad(sample['segmentations'])

        return sample"""
        padding = self.padding
        if padding is None:
            s, h_i, w_i, c = sample['point_clouds'].shape
            h_o, w_o = self.output_shape
            h_pl, w_pl = (h_o - h_i) // 2, (w_o - w_i) // 2
            h_pr, w_pr = h_o - h_i - h_pl, w_o - w_i - w_pl
            padding = (0, 0), (h_pl, h_pr), (w_pl, w_pr), (0, 0)

        sample['point_clouds'] = np.pad(sample['sequences'], padding)
        return sample
