import torch
import os
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from time import time


class Trainer:
    def __init__(self, reduction, network, training_dataloader,
                 validation_dataloader, optimiser, 
                 scheduler=None, experiment_name='CenterNet',
                 epoch=None, save_states=True):
        self.save_states = save_states
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
            
        self.experiment_name = experiment_name
        self.joints = network.joints

        self.network = network
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.writer = SummaryWriter(os.path.join('runs', experiment_name))
        self.reduction = reduction

        if epoch is None:
            self.epoch = 0
        else:
            self.epoch = epoch
            self.load_state()
            self.epoch += 1 # start next epoch

    def load_state(self):
        part = self.epoch + 1
        model_path = os.path.join('models', self.experiment_name, f'net_{part}.pt')   
        self.network.load_state_dict(torch.load(model_path))

        optimiser_path = os.path.join('models', self.experiment_name, f'optimiser_{part}.pt')   
        self.optimiser.load_state_dict(torch.load(optimiser_path))

        if self.scheduler:
            scheduler_path = os.path.join('models', self.experiment_name, f'scheduler_{part}.pt')   
            self.scheduler.load_state_dict(torch.load(scheduler_path))

    def train(self, epochs):
        for e in range(self.epoch, epochs):
            print(f"Training: \nEpoch {e+1}\n-------------------------------")
            start = time()           
            tr_loss, tr_acc = self.train_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average training loss: {tr_loss}, Average training MPJPE: {tr_acc}")
            print(f"Validation: \nEpoch {e+1}\n-------------------------------")
            start = time()       
            val_loss, val_acc = self.validation_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average validation loss: {val_loss}, Average validation MPJPE: {val_acc}")
            self.log_loss(tr_loss, val_loss, e + 1)
            self.log_accuracy(tr_acc, val_acc, e + 1)

            if (self.save_states):
                self.save(e + 1)
        self.writer.flush()

        self.writer.close()

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
            heat_maps_targets = batched_sample['heat_maps']
            target_offsets = batched_sample['2D_skeletons_offsets']
            target_depths = batched_sample['2D_skeletons_depths'] / 1000
            skeletons_targets = batched_sample['skeletons']      
            b, j, n = skeletons_targets.shape
            if b < 2:
                continue
      
            heat_maps, offsets, depths = self.network(sequences)
            loss = self.network.loss(heat_maps,
                                     offsets,
                                     depths,
                                     heat_maps_targets,
                                     target_offsets,
                                     target_depths)
            if torch.any(loss.isnan()):
                print('nan occured in loss')
                continue
            self.optimise(loss)
            
            predicted_3D_skeleton = predict_3D_skeleton(heat_maps, offsets, depths,
                                                            batched_sample['intrinsic_matrix_inverted'],
                                                            batched_sample['rotation_matrix_inverted'],
                                                            batched_sample['translation_vector_inverted'], 4)
            acc = MPJPE(predicted_3D_skeleton, skeletons_targets)
            
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
                heat_maps_targets = batched_sample['heat_maps']
                target_offsets = batched_sample['2D_skeletons_offsets']
                target_depths = batched_sample['2D_skeletons_depths'] / 1000
                skeletons_targets = batched_sample['skeletons']   
                b, j, n = skeletons_targets.shape
                if b < 2:
                    continue
          
                heat_maps, offsets, depths = self.network(sequences)
                loss = self.network.loss(heat_maps,
                                         offsets,
                                         depths,
                                         heat_maps_targets,
                                         target_offsets,
                                         target_depths)

                if torch.any(loss.isnan()):
                    print('nan occured in loss')
                    continue
                predicted_3D_skeleton = predict_3D_skeleton(heat_maps, offsets, depths,
                                                            batched_sample['intrinsic_matrix_inverted'],
                                                            batched_sample['rotation_matrix_inverted'],
                                                            batched_sample['translation_vector_inverted'], 4)
                acc = MPJPE(predicted_3D_skeleton, skeletons_targets)
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
          
    def log_loss(self, training_loss, validation_loss, epoch):  
        self.writer.add_scalar('average training loss', training_loss, epoch)
        self.writer.add_scalar('average validation loss', validation_loss, epoch)
        
    def log_accuracy(self, training_accuracy, validation_accuracy, epoch): 
        self.writer.add_scalar('average training MPJPE', training_accuracy, epoch)
        self.writer.add_scalar('average validation MPJPE', validation_accuracy, epoch)

    def optimise(self, loss : torch.Tensor):
        if torch.any(loss.isnan()):
            print('nan occured in loss')
            return
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def save(self, epoch):
        path = os.path.join('models', self.experiment_name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        torch.save(self.network.state_dict(), os.path.join(path, f'net_{epoch}.pt'))
        torch.save(self.optimiser.state_dict(), os.path.join(path, f'optimiser_{epoch}.pt'))
        if self.scheduler:
            torch.save(self.scheduler.state_dict(), os.path.join(path, f'scheduler_{epoch}.pt'))
            
def rescale(pt_cloud, mean, min_, max_, a=-1, b=1):
    if len(pt_cloud.shape) == 3:
        pt_cloud = (pt_cloud - a) * (max_[None, :, None] - min_[None, :, None]) / (b - a)
        pt_cloud += min_[None, :, None]
        pt_cloud += mean[None, :, None]
    elif len(pt_cloud.shape) == 4:
        pt_cloud = (pt_cloud - a) * (max_[None, :, None, None] - min_[None, :, None, None]) / (b - a)
        pt_cloud += min_[None, :, None, None]
        pt_cloud += mean[None, :, None, None]
    return pt_cloud

def MPJPE(prediction, ground_truth):
    return torch.mean(torch.sqrt(torch.sum((prediction - ground_truth) ** 2, 1)))

def predict_3D_skeleton(heat_maps, offsets, depths,
                        intrinsic_matrix_inverted,
                        rotation_matrix_inverted,
                        translation_vector_inverted,
                        stride=4):
        b, j, h, w = heat_maps.shape
        heat_maps_flattened = torch.flatten(heat_maps, 2)
        idx = torch.argmax(heat_maps_flattened, 2)[:, :, None]
        xs = idx % w
        ys = idx // w
        positions_2D = torch.stack((xs, ys), 1).squeeze(-1)

        offsets_flattened = torch.flatten(offsets, 2)
        x_offs = offsets_flattened[:, 0, None, :].expand(-1, j, -1)
        y_offs = offsets_flattened[:, 1, None, :].expand(-1, j, -1)
        relevant_x_offs = torch.take_along_dim(x_offs, idx, 2).squeeze(-1)
        relevant_y_offs = torch.take_along_dim(y_offs, idx, 2).squeeze(-1)
        relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 1)

        depths_flattened = -torch.flatten(depths, 2) * 1000
        relevant_depths = torch.take_along_dim(depths_flattened, idx, 2).squeeze(-1)[:, None, :]
        positions_corrected = (positions_2D + relevant_offsets) * stride

        points = torch.concatenate((positions_corrected, torch.ones(b, 1, j, device=positions_corrected.device)), 1)
        points *= relevant_depths
        """points_3D = torch.bmm(rotation_matrix_inverted.expand(b, 3, 3),
                              translation_vector_inverted[None, :, None] + \
                              torch.bmm(intrinsic_matrix_inverted.expand(b, 3, 3), points))"""
        points_flat = torch.reshape(torch.moveaxis(points, 1, 0), (3, b * j))
        points_3D_flat = torch.matmul(rotation_matrix_inverted, torch.matmul(intrinsic_matrix_inverted, points_flat) + translation_vector_inverted[:, None])
        points_3D = torch.moveaxis(torch.reshape(points_3D_flat, (3, b, j)), 0, 1)
        return points_3D
