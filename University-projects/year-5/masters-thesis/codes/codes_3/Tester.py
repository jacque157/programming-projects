import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

from DatasetLoader import DatasetLoader
from utils import plot_body, plot_skeleton
JOINTS = 22

class RelativePoseTester:
    def __init__(self, network : nn.Module, testing_dataloader : DatasetLoader, 
                 mean, min_, max_, a=torch.tensor(-1), b=torch.tensor(1)):      
        self.mean = mean
        self.min_ = min_
        self.max_ = max_
        self.a = a
        self.b = b
        self.testing_dataloader = testing_dataloader
        
        self.joints = network.joints

        self.network = network
        self.accuracy_function = self.get_accuracy_function()
    
    def get_accuracy_function(self) -> nn.Module:
        def acc(relative_joints, relative_target_joints):
            relative_joints = rescale(relative_joints, self.mean, self.min_, self.max_, self.a, self.b)
            relative_target_joints = rescale(relative_target_joints, self.mean, self.min_, self.max_, self.a, self.b)
            return torch.mean(torch.sqrt(torch.sum(torch.square(relative_joints - relative_target_joints), axis=1)))      
        return acc
    
    def test(self, plot_predictions=False):
        acc = self.test_loop(plot_predictions)
        print(f"Average testing MPJPE: {acc}")
    
    def test_loop(self, plot_predictions):
        self.network.eval()
        accuracy_overall = 0
        samples_count = 0

        with torch.no_grad():
            for i, batched_sample in enumerate(self.testing_dataloader):              
                sequences = batched_sample['sequences']
                b, c, h, w = sequences.shape
                skeletons_targets  = batched_sample['key_points']
                skeletons_targets = skeletons_targets[:, :, 1:] # ommit root joint, the relative root joint is always [0, 0, 0]   
                if b < 2:
                    continue
        
                predicted_joints = self.network(sequences)        
                acc = self.accuracy(predicted_joints, skeletons_targets)

                samples_count += 1
                sample_acc = acc.item()
                torch.cuda.empty_cache()
                
                if plot_predictions:
                    self.plot(batched_sample, predicted_joints)

                accuracy_overall += sample_acc

                if i % 20 == 0:
                    name = batched_sample['name']
                    print(f'dataset: {name}, batch: {i}, ' +
                            f'relative MPJPE: {sample_acc:.03f}, ')
   
        return accuracy_overall / samples_count
    
    def accuracy(self, joints_prediction : torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        return self.accuracy_function(joints_prediction, joints)
    
    def plot(self, sample, predictions):
        sequences = sample['sequences']
        b, c, h, w = sequences.shape
        skeletons_targets  = sample['key_points']
        roots = sample['root_key_points'][:, :, None]

        point_clouds = sequences.reshape(b, c, h * w)
        scaled_sequences = rescale(point_clouds, self.mean, self.min_, self.max_, self.a, self.b).reshape(b, c, h, w)
        scaled_skeletons = rescale(skeletons_targets + roots, self.mean, self.min_, self.max_, self.a, self.b)
        predicted_joints = torch.cat((torch.zeros(b, 3, 1, device=roots.device), predictions), 2)
        scaled_predictions = rescale(predicted_joints + roots, self.mean, self.min_, self.max_, self.a, self.b)
        
        for j in range(b):
            scaled_image = np.moveaxis(scaled_sequences[j].detach().cpu().numpy(), 0, -1)
            scaled_skeleton = np.moveaxis(scaled_skeletons[j].detach().cpu().numpy(), 0, -1)
            scaled_prediction = np.moveaxis(scaled_predictions[j].detach().cpu().numpy(), 0, -1)

            ax = plot_body(scaled_image)
            ax.set_title('Ground Truth')
            plot_skeleton(scaled_skeleton, ax)

            ax = plot_body(scaled_image)
            ax.set_title('Prediction')
            plot_skeleton(scaled_prediction, ax)
            
            plt.show()

"""class AbsolutePosTester(RelativePoseTester):
    def __init__(self, network : nn.Module, testing_dataloader : DatasetLoader, 
                 mean, min_, max_, a=torch.tensor(-1), b=torch.tensor(1)):   
        super().__init__(network, testing_dataloader, mean, min_, max_, a, b)

    def test_loop(self):
        self.network.eval()
        accuracy_overall = 0
        samples_count = 0

        with torch.no_grad():
            for i, batched_sample in enumerate(self.training_dataloader):              
                sequences = batched_sample['sequences']
                b, c, h, w = sequences.shape
                skeletons_targets  = batched_sample['key_points']

                if b < 2:
                    continue
        
                predicted_joints = self.network(sequences)       
                acc = self.accuracy(predicted_joints, skeletons_targets)

                samples_count += 1
                sample_acc = acc.item()
                torch.cuda.empty_cache()
                
                accuracy_overall += sample_acc

                if i % 20 == 0:
                    name = batched_sample['name']
                    print(f'dataset: {name}, batch: {i}, ' +
                            f'absolute MPJPE: {sample_acc:.03f}, ')
   
        return accuracy_overall / samples_count"""


class PointNetRelativePoseTester(RelativePoseTester):
    def __init__(self, network : nn.Module, testing_dataloader : DatasetLoader, 
                 mean, min_, max_, a=torch.tensor(-1), b=torch.tensor(1)):   
        super().__init__(network, testing_dataloader, mean, min_, max_, a, b)

    def test_loop(self, plot_predictions):
        self.network.eval()
        accuracy_overall = 0
        samples_count = 0

        with torch.no_grad():
            for i, batched_sample in enumerate(self.testing_dataloader):    
                          
                sequences = batched_sample['sequences']           
                skeletons_targets  = batched_sample['key_points']
                skeletons_targets = skeletons_targets[:, :, 1:] 
                b, j, n = skeletons_targets.shape

                if b < 2:
                    continue
        
                m3x3, m64x64, predicted_joints = self.network(sequences)     
                acc = self.accuracy(predicted_joints, skeletons_targets)

                samples_count += 1
                sample_acc = acc.item()
                torch.cuda.empty_cache()
                if plot_predictions:
                    self.plot(batched_sample, predicted_joints)

                accuracy_overall += sample_acc

                if i % 20 == 0:
                    name = batched_sample['name']
                    print(f'dataset: {name}, batch: {i}, ' +
                            f'absolute MPJPE: {sample_acc:.03f}, ')
   
        return accuracy_overall / samples_count


    def plot(self, sample, predictions):
        point_clouds = sample['sequences']
        b, c, n = point_clouds.shape
        skeletons_targets  = sample['key_points']
        roots = sample['root_key_points'][:, :, None]

        scaled_sequences = rescale(point_clouds, self.mean, self.min_, self.max_, self.a, self.b).reshape(b, c, 2, n // 2)
        scaled_skeletons = rescale(skeletons_targets + roots, self.mean, self.min_, self.max_, self.a, self.b)
        predicted_joints = torch.cat((torch.zeros(b, 3, 1, device=roots.device), predictions), 2)
        scaled_predictions = rescale(predicted_joints + roots, self.mean, self.min_, self.max_, self.a, self.b)
        
        for j in range(b):
            scaled_image = np.moveaxis(scaled_sequences[j].detach().cpu().numpy(), 0, -1)
            scaled_skeleton = np.moveaxis(scaled_skeletons[j].detach().cpu().numpy(), 0, -1)
            scaled_prediction = np.moveaxis(scaled_predictions[j].detach().cpu().numpy(), 0, -1)

            ax = plot_body(scaled_image)
            ax.set_title('Ground Truth')
            plot_skeleton(scaled_skeleton, ax)

            ax = plot_body(scaled_image)
            ax.set_title('Prediction')
            plot_skeleton(scaled_prediction, ax)
            
            plt.show()

def rescale(pt_cloud, mean, min_, max_, a=-1, b=1):
    pt_cloud = (pt_cloud - a) * (max_[None, :, None] - min_[None, :, None]) / (b - a)
    pt_cloud += min_[None, :, None]
    pt_cloud += mean[None, :, None]
    return pt_cloud