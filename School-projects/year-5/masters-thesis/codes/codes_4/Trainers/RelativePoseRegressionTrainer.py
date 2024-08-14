import torch
import os
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from time import time
from Trainers.RegressionTrainer import Trainer as RegressionTrainer
from Trainers.RegressionTrainer import project_camera_skeletons, rescale, MPJPE

class Trainer(RegressionTrainer):
    def __init__(self, network, training_dataloader,
                 validation_dataloader, optimiser, 
                 scheduler=None, experiment_name='RelativePoseRegression',
                 epoch=None, save_states=True):
        super().__init__(network, training_dataloader,
                 validation_dataloader, optimiser, 
                 scheduler, experiment_name,
                 epoch, save_states)
        self.device = None

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
            loss = self.network.loss(predicted_skeleton, relative_pose(skeletons_targets))
            predicted_skeleton = absolute_pose(predicted_skeleton)
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
                loss = self.network.loss(predicted_skeleton, relative_pose(skeletons_targets))
                predicted_skeleton = absolute_pose(predicted_skeleton)
                
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
#                            0,    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
_PARENTS = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4, 5, 6,  7,  8,  9,   9,  9, 12, 13, 14, 16, 17, 18, 19])
                    
def relative_pose(absolute_skeletons):
    parents = torch.tensor([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
                           device=absolute_skeletons.device)
    relative_skeletons = absolute_skeletons.clone()
    relative_skeletons[:, :, 1:] = absolute_skeletons[:, :, 1:] - absolute_skeletons[:, :, parents[1:]]
    return relative_skeletons

def absolute_pose(relative_skeletons):
    absolute_skeletons = relative_skeletons.clone()
    absolute_skeletons[:, :, [1, 2, 3]] += absolute_skeletons[:, :, 0, None]
    absolute_skeletons[:, :, [4, 5, 6]] += absolute_skeletons[:, :, [1, 2, 3]]
    absolute_skeletons[:, :, [7, 8, 9]] += absolute_skeletons[:, :, [4, 5, 6]]
    absolute_skeletons[:, :, [10, 11, 12, 13, 14]] += absolute_skeletons[:, :, [7, 8, 9, 9, 9]]
    absolute_skeletons[:, :, [15, 16, 17]] += absolute_skeletons[:, :, [12, 13, 14]]
    absolute_skeletons[:, :, [18, 19]] += absolute_skeletons[:, :, [16, 17]]
    absolute_skeletons[:, :, [20, 21]] += absolute_skeletons[:, :, [18, 19]]
    return absolute_skeletons
    
