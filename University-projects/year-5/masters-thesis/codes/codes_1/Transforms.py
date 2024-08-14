import torch
from torch import nn
import numpy as np

# {'sequences' : poses, 'valid_points' : mask, 'key_points' : skeletons, 'root_keypoints' : centres}

class ZeroCenter(object):
    def __init__(self, mean : np.array = None):
        self.mean = mean

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        point_clouds = sample['sequences']
        skelybums = sample['key_points']
        
        if self.mean is not None:
            point_clouds_centered = point_clouds - self.mean
            skelybums -= self.mean
        else:
            mean = np.mean(point_clouds, axis=(1, 2))
            point_clouds_centered = point_clouds - mean[:, np.newaxis, np.newaxis, :]
            skelybums -= mean[:, np.newaxis, :]

        sample['sequences'] = point_clouds_centered
        sample['key_points'] = skelybums
        sample['root_key_points'] = sample['key_points'][:, 0]
        return sample
    
class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample : dict[str, np.array]) -> dict[str, torch.Tensor]:
        point_clouds = np.moveaxis(sample['sequences'], 3, 1)
        skeletons = np.moveaxis(sample['key_points'], 2, 1) 
        
        sample['sequences'] = torch.from_numpy(point_clouds).to(torch.float32)
        sample['valid_points'] = torch.from_numpy(sample['valid_points']).to(torch.bool)
        sample['key_points'] = torch.from_numpy(skeletons).to(torch.float32)
        sample['root_key_points'] = torch.from_numpy(sample['root_key_points']).to(torch.float32)
        if 'segmentations' in sample:
            sample['segmentations'] = torch.from_numpy(sample['segmentations']).to(torch.long)
        return sample
    
class ToNumpy(object):
    def __init__(self):
        pass

    def __call__(self, sample : dict[str, torch.Tensor]) -> dict[str, np.array]:
        point_clouds = torch.movedim(sample['sequences'], 1, 3)
        skeletons = torch.movedim(sample['key_points'], 1, 2) 
        
        sample['sequences'] = point_clouds.numpy()
        sample['valid_points'] = sample['valid_points'].numpy()
        sample['key_points'] = skeletons.numpy()
        sample['root_key_points'] = sample['root_key_points'].numpy()
        if 'segmentations' in sample:
            sample['segmentations'] = sample['segmentations'].numpy()
        return sample

class Rescale(object):
    def __init__(self, min_ : np.array, max_ : np.array, a : int = -1, b : int = 1):
        self.min = min_
        self.max = max_
        self.a = a
        self.b = b

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        point_clouds = sample['sequences']
        skelybums = sample['key_points']

        point_clouds_scaled = self.a + ((point_clouds - self.min) * (self.b - self.a) / (self.max - self.min))
        skelybums_scaled = self.a + ((skelybums - self.min) * (self.b - self.a) / (self.max - self.min))
        sample['sequences'] = point_clouds_scaled
        sample['key_points'] = skelybums_scaled
        sample['root_key_points'] = sample['key_points'][:, 0]
        return sample

class RandomCrop(object):
    def __init__(self, output_shape : tuple[int, int], input_shape : tuple[int, int] = None):
        self.output_shape = output_shape
        if input_shape is None:
            self.high_w = None
            self.high_h = None
        else:
            h0, w0 = input_shape
            h1, w1, = output_shape
            self.high_w = w0 - w1
            self.high_h = h0 - h1      

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        point_clouds = sample['sequences']
        masks = sample['valid_points']
        if 'segmentations' in sample:
            segmentations = sample['segmentations']
        h1, w1, = self.output_shape   

        if self.high_w is None or self.high_h is None:
            _, h1, w1, _ = point_clouds.size()
        else:
            high_w = self.high_w
            high_h = self.high_h
        offset_x = np.random.randint(0, high_w)
        offset_y = np.random.randint(0, high_h)
        cropped_point_clouds = point_clouds[:, offset_y : offset_y + h1, offset_x : offset_x + w1, :]
        cropped_masks = masks[:, offset_y : offset_y + h1, offset_x : offset_x + w1]
        if 'segmentations' in sample:
            cropped_segmentations = segmentations[:, offset_y : offset_y + h1, offset_x : offset_x + w1]
        
        sample['sequences'] = cropped_point_clouds
        sample['valid_points'] = cropped_masks
        if 'segmentations' in sample:
            sample['segmentations'] = cropped_segmentations
        return sample

class Crop(object):
    def __init__(self, output_shape : tuple[int, int], input_shape : tuple[int, int] = None):
        self.output_shape = output_shape
        if input_shape is None:
            self.offset = None
        else:
            h0, w0 = input_shape
            h1, w1, = output_shape

            w_offset = (w0 - w1) // 2
            h_offset = (h0 - h1) // 2
            self.offset = (h_offset, w_offset)
  

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        point_clouds = sample['sequences']
        masks = sample['valid_points']
        if 'segmentations' in sample:
            segmentations = sample['segmentations']
        h1, w1, = self.output_shape   

        offset = self.offset

        if offset is None:
            _, h0, w0, _ = point_clouds.shape
            w_offset = (w0 - w1) // 2
            h_offset = (h0 - h1) // 2
            offset = (h_offset, w_offset)

        cropped_point_clouds = point_clouds[:, h_offset : h_offset + h1, w_offset : w_offset + w1, :]
        cropped_masks = masks[:, h_offset : h_offset + h1, w_offset : w_offset + w1]
        if 'segmentations' in sample:
            cropped_segmentations = segmentations[:, h_offset : h_offset + h1, w_offset : w_offset + w1]
        
        sample['sequences'] = cropped_point_clouds
        sample['valid_points'] = cropped_masks
        if 'segmentations' in sample:
            sample['segmentations'] = cropped_segmentations
        return sample

class ZeroOutEntries(object):
    def __init__(self):
        pass

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        point_clouds = sample['sequences'] 
        masks = sample['valid_points']

        point_clouds[masks == False] = np.zeros(3)

        sample['sequences'] = point_clouds     
        return sample
    
class RelativeJointsPosition(object):
    def __init__(self):
        self.joints_id = [11, 8, 5, 2, 0, # right leg
                          10, 7, 4, 1, 0, # left leg
                          0, 3, 6, 9, 12, 15, # spine
                          21, 19, 17, 14, # right arm
                          20, 18, 16, 13] # left arm
        self.parents_id= [8, 5, 2, 0, 0,
                          7, 4, 1, 0, 0,
                          0, 0, 3, 6, 9, 12,
                          19, 17, 14, 9,
                          18, 16, 13, 9]
        
    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        skeletons = sample['key_points'] 
        
        pelvis = skeletons[:, 0, :]
        skeletons[:, self.joints_id, :] = skeletons[:, self.joints_id, :] - skeletons[:, self.parents_id, :]
        skeletons[:, 0, :] = pelvis

        sample['key_points'] = skeletons
        return sample
        

class ToDevice(object):
    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, sample : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample['sequences'] = sample['sequences'].to(device=self.device)
        sample['valid_points'] = sample['valid_points'].to(device=self.device)
        sample['key_points'] = sample['key_points'].to(device=self.device)
        sample['root_key_points'] = sample['root_key_points'].to(device=self.device)
        if 'segmentations' in sample:
            sample['segmentations'] = sample['segmentations'].to(device=self.device)

        return sample


class AutoGrad(object):
    def __init__(self, requires_grad=True):
        self.grad = requires_grad

    def __call__(self, sample : dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample['sequences'] = sample['sequences'].requires_grad_(self.grad)
        #sample['valid_points'] = sample['valid_points'].requires_grad_(self.grad)
        sample['key_points'] = sample['key_points'].requires_grad_(self.grad)
        sample['root_key_points'] = sample['root_key_points'].requires_grad_(self.grad)
        #if 'segmentations' in sample:
            #sample['segmentations'] = sample['segmentations'].requires_grad_(self.grad)

        return sample
    
class Shuffle(object):
    def __init__(self):
        pass

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        s, h, w, c = sample['sequences'].shape
        idx = np.random.choice(s, s, replace=False)
        sample['sequences'] = sample['sequences'][idx]
        sample['valid_points'] = sample['valid_points'][idx]
        sample['key_points'] = sample['key_points'][idx]
        sample['root_key_points'] = sample['root_key_points'][idx]
        if 'segmentations' in sample:
            sample['segmentations'] = sample['segmentations'][idx]

        return sample
    
class ZeroPad(object):
    def __init__(self, output_shape : tuple[int, int], input_shape : tuple[int, int] = None):    
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
            s, h_i, w_i, c = sample['sequences'].shape
            h_o, w_o = self.output_shape
            h_pl, w_pl = (h_o - h_i) // 2, (w_o - w_i) // 2
            h_pr, w_pr = h_o - h_i - h_pl, w_o - w_i - w_pl
            padding = (0, 0), (h_pl, h_pr), (w_pl, w_pr), (0, 0)

        sample['sequences'] = np.pad(sample['sequences'], padding)
        if 'segmentations' in sample:
            sample['segmentations'] = np.pad(sample['segmentations'], padding[:-1])

        return sample

class RootAlignedPose(object):
    def __init__(self, mean : np.array = None):
        self.mean = mean

    def __call__(self, sample : dict[str, np.array]) -> dict[str, np.array]:
        poses = sample['key_points']
        roots = poses[:, 0]
        
        poses = poses - roots[:, None, :]
        
        sample['key_points'] = poses
        return sample
        

# sample = {'sequences' : poses, 'valid_points' : mask, 'key_points' : skeletons, 'root_key_points' : centres}