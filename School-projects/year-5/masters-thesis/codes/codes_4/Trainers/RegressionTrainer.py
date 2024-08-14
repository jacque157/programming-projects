import torch
import os
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from time import time
from Trainers.CenterNetTrainer import Trainer as CenterNetTrainer


class Trainer(CenterNetTrainer):
    def __init__(self, network, training_dataloader,
                 validation_dataloader, optimiser, 
                 scheduler=None, experiment_name='PoseRegression',
                 epoch=None, save_states=True):
        super().__init__(None, network, training_dataloader,
                 validation_dataloader, optimiser, 
                 scheduler, experiment_name,
                 epoch, save_states)

    def train_loop(self):
        self.network.train()
        loss_overall = 0
        accuracy_overall = 0
        samples_count = 0
        for i, batched_sample in enumerate(self.training_dataloader):
            if 'point_clouds' in batched_sample:
                sequences = batched_sample['point_clouds']
            else:
                sequences = batched_sample['images']
            skeletons_targets = batched_sample['skeletons']     
            b, j, n = skeletons_targets.shape
            if b < 2:
                continue
      
            predicted_skeleton = self.network(sequences)
            loss = self.network.loss(predicted_skeleton, skeletons_targets)
            
            if torch.any(loss.isnan()):
                print('nan occured in loss')
                continue
            self.optimise(loss)
            
            predicted_skeleton = rescale(predicted_skeleton, batched_sample['center'] ,
                                         batched_sample['min_'], batched_sample['max_'],
                                         a=batched_sample['a'], b=batched_sample['b'])
            predicted_skeleton_3D = project_camera_skeletons(predicted_skeleton,
                                                             batched_sample['rotation_matrix_inverted'].double(),
                                                             batched_sample['translation_vector_inverted'].double())

            skeletons_targets = rescale(skeletons_targets, batched_sample['center'] ,
                                        batched_sample['min_'], batched_sample['max_'],
                                        a=batched_sample['a'], b=batched_sample['b'])
            skeletons_targets_3D = project_camera_skeletons(skeletons_targets,
                                                            batched_sample['rotation_matrix_inverted'].double(),
                                                            batched_sample['translation_vector_inverted'].double())

            acc = MPJPE(predicted_skeleton_3D, skeletons_targets_3D)
            
            samples_count += 1
            sample_loss = loss.item()
            sample_acc = acc.item()
            torch.cuda.empty_cache()
            loss_overall += sample_loss
            accuracy_overall += sample_acc

            if i % 20 == 0:
                name = batched_sample['name']
                print(f'dataset: {name}, batch: {i}, ' +
                        f'loss: {sample_loss:.03f}, ' +
                        f'absolute MPJPE: {sample_acc:.03f}, ')

        if self.scheduler:
            self.scheduler.step()
        
        return loss_overall / samples_count, accuracy_overall / samples_count

    def validation_loop(self):
        self.network.eval()
        loss_overall = 0
        accuracy_overall = 0
        samples_count = 0

        with torch.no_grad():
            for i, batched_sample in enumerate(self.validation_dataloader):                           
                if 'point_clouds' in batched_sample:
                    sequences = batched_sample['point_clouds']
                else:
                    sequences = batched_sample['images']
                skeletons_targets = batched_sample['skeletons']     
                b, j, n = skeletons_targets.shape
                if b < 2:
                    continue
          
                predicted_skeleton = self.network(sequences)
                loss = self.network.loss(predicted_skeleton, skeletons_targets)
                
                if torch.any(loss.isnan()):
                    print('nan occured in loss')
                    continue
                predicted_skeleton = rescale(predicted_skeleton, batched_sample['center'] ,
                                         batched_sample['min_'], batched_sample['max_'],
                                         a=batched_sample['a'], b=batched_sample['b'])
                predicted_skeleton_3D = project_camera_skeletons(predicted_skeleton,
                                                                 batched_sample['rotation_matrix_inverted'].double(),
                                                                 batched_sample['translation_vector_inverted'].double())

                skeletons_targets = rescale(skeletons_targets, batched_sample['center'] ,
                                            batched_sample['min_'], batched_sample['max_'],
                                            a=batched_sample['a'], b=batched_sample['b'])
                skeletons_targets_3D = project_camera_skeletons(skeletons_targets,
                                                                batched_sample['rotation_matrix_inverted'].double(),
                                                                batched_sample['translation_vector_inverted'].double())

                acc = MPJPE(predicted_skeleton_3D, skeletons_targets_3D)
                samples_count += 1
                sample_loss = loss.item()
                sample_acc = acc.item()
                torch.cuda.empty_cache()
                
                loss_overall += sample_loss
                accuracy_overall += sample_acc

                if i % 20 == 0:
                    name = batched_sample['name']
                    print(f'dataset: {name}, batch: {i}, ' +
                            f'loss: {sample_loss:.03f}, ' +
                            f'absolute MPJPE: {sample_acc:.03f}, ')
   
        return loss_overall / samples_count, accuracy_overall / samples_count

                    
def rescale(skeletons, mean, min_, max_, a=-1, b=1):
    skeletons = (skeletons - a) * (max_[None, :, None] - min_[None, :, None]) / (b - a)
    skeletons += min_[None, :, None]
    skeletons += mean[None, :, None]
    return skeletons

def project_camera_skeletons(skeletons, rotation_matrix_inverted, translation_vector_inverted):
    points_3D = torch.matmul(rotation_matrix_inverted, skeletons + translation_vector_inverted[None, :, None])
    return points_3D
    
def MPJPE(prediction, target):
    return torch.mean(torch.sqrt(torch.sum((prediction - target) ** 2, 1))) 
