import torch
import os
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from time import time

from Dataset import Poses3D
from utils import reconstruct_skeletons
from DatasetLoader import DatasetLoader

JOINTS = 22

class Trainer:
    def __init__(self, network : nn.Module, training_datasets : list[Poses3D],
                 validation_datasets : list[Poses3D], optimiser : torch.optim.Adam, 
                 scheduler : torch.optim.lr_scheduler.LinearLR=None,
                 experiment_name='experiment', batch_size=32, epoch=None,
                 relative_joints=False):
        
        self.training_datasets = training_datasets
        self.training_dataloaders = [data.DataLoader(training_dataset,
                                                     batch_size=1,
                                                     shuffle=True) for training_dataset in self.training_datasets]
        self.validation_datasets = validation_datasets
        self.validation_dataloaders = [data.DataLoader(validation_dataset,
                                                       batch_size=1) for validation_dataset in self.validation_datasets]
        
        self.experiment_name = experiment_name
        self.relative_joints = relative_joints
        self.joints = network.joints

        self.network = network
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.writer = SummaryWriter(os.path.join('runs', experiment_name))

        if epoch is None:
            self.epoch = 0
        else:
            self.epoch = epoch
            self.load_state()
            self.epoch += 1 # start next epoch

        self.loss_function = self.get_loss_function()
        self.accuracy_function = self.get_accuracy_function()
        self.batch_size = batch_size

    def load_state(self):
        part = self.epoch + 1
        model_path = os.path.join('models', self.experiment_name, f'net_{part}.pt')   
        self.network.load_state_dict(torch.load(model_path))

        optimiser_path = os.path.join('models', self.experiment_name, f'optimiser_{part}.pt')   
        self.optimiser.load_state_dict(torch.load(optimiser_path))

        if self.scheduler:
            scheduler_path = os.path.join('models', self.experiment_name, f'scheduler_{part}.pt')   
            self.scheduler.load_state_dict(torch.load(scheduler_path))

    def get_loss_function(self) -> nn.Module:
        """def loss(sample, target):
            return torch.mean(torch.sqrt(torch.sum(torch.square(sample - target), axis=1)))
        return loss #nn.MSELoss(reduction='sum')"""
        return nn.MSELoss(reduction='mean')
    
    def get_accuracy_function(self) -> nn.Module:
        def acc(sample, target):
            return torch.mean(torch.sqrt(torch.sum(torch.square(sample - target), axis=1)))
        
        def rel_acc(sample_relative, target_relative):
            s, c, j = sample_relative.shape

            sample = sample_relative.view(s, j, c)
            sample = reconstruct_skeletons(sample)
            sample = sample.view(s, c, j)

            target = target_relative.view(s, j, c)
            target = reconstruct_skeletons(target)
            target = target.view(s, c, j)

            return torch.mean(torch.sqrt(torch.sum(torch.square(sample - target), axis=1)))
        
        return rel_acc if self.relative_joints else acc

    def train(self, epochs):
        for e in range(self.epoch, epochs):
            print(f"Training: \nEpoch {e+1}\n-------------------------------")
            start = time()           
            tr_loss, tr_acc = self.train_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average training loss: {tr_loss}, Average training accuracy: {tr_acc}")
            print(f"Validation: \nEpoch {e+1}\n-------------------------------")
            start = time()       
            val_loss, val_acc = self.validation_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average validation loss: {val_loss}, Average validation accuracy: {val_acc}")
            self.log_loss(tr_loss, val_loss, e + 1)
            self.log_accuracy(tr_acc, val_acc, e + 1)

            self.save(e + 1)
        self.writer.flush()

        self.writer.close()

    def train_loop(self):
        self.network.train()
        loss_overall = 0
        accuracy_overall = 0
        samples_count = 0
        for i, dataloader in enumerate(self.training_dataloaders):
            size = len(dataloader.dataset)
            
            for j, sample in enumerate(dataloader):     
                sequences = torch.squeeze(sample['sequences'], 0)
                s, c, h, w = sequences.shape
                targets  = torch.squeeze(sample['key_points'], 0)

                sample_loss = 0
                sample_acc = 0
                samples = 0
                for k in range(0, s, self.batch_size):
                    sequence = sequences[k:k+self.batch_size]
                    key_points = targets[k:k+self.batch_size]
                    if len(sequence) < 2:
                        continue
                    samples_count += 1
                    samples += 1

                    prediction = self.network(sequence)
                    loss = self.loss(prediction, key_points)
                    self.optimise(loss)
                    acc = self.accuracy(prediction, key_points)

                    sample_loss += loss.item()
                    sample_acc += acc.item()
                    loss_overall += loss.item()
                    accuracy_overall += acc.item()

                if j % 10 == 0 and samples > 0:
                    print(f'dataset: {dataloader.dataset.name}, {j}/{size}, loss: {sample_loss / samples}, accuracy: {sample_acc / samples}')

        if self.scheduler:
            self.scheduler.step()
        
        return loss_overall / samples_count, accuracy_overall / samples_count

    def validation_loop(self):
        self.network.eval()
        loss_overall = 0
        accuracy_overall = 0
        samples_count = 0
        with torch.no_grad():
            for i, dataloader in enumerate(self.validation_dataloaders):
                size = len(dataloader.dataset)
                
                sample_loss = 0
                sample_acc = 0
                samples = 0

                for j, sample in enumerate(dataloader):     
                    sequences = torch.squeeze(sample['sequences'], 0)
                    s, c, h, w = sequences.shape
                    targets  = torch.squeeze(sample['key_points'], 0)
                    for k in range(0, s, self.batch_size):
                        sequence = sequences[k:k+self.batch_size]
                        key_points = targets[k:k+self.batch_size]
                        if len(sequence) < 2:
                            continue
                        samples_count += 1
                        samples += 1
                        
                        prediction = self.network(sequence)
                        loss = self.loss(prediction, key_points)
                        acc = self.accuracy(prediction, key_points)

                        sample_loss += loss.item()
                        sample_acc += acc.item()
                        loss_overall += loss.item()
                        accuracy_overall += acc.item()
                    if j % 10 == 0 and samples > 0:
                        print(f'dataset: {dataloader.dataset.name}, {j}/{size}, loss: {sample_loss / samples}, accuracy: {sample_acc / samples}')

        return loss_overall / samples_count, accuracy_overall / samples_count

    def loss(self, prediction : torch.Tensor, sample : dict[str, torch.Tensor]) -> torch.Tensor:
        j = self.network.joints
        if type(sample) is dict:
            ground_truth = torch.squeeze(sample['key_points'], 0)
        else:
            ground_truth = sample
        return self.loss_function(prediction, ground_truth)
    
    def accuracy(self, prediction : torch.Tensor, sample : dict[str, torch.Tensor]) -> torch.Tensor:
        j = self.network.joints
        if type(sample) is dict:
            ground_truth = torch.squeeze(sample['key_points'], 0)
        else:
            ground_truth = sample
        return self.accuracy_function(prediction, ground_truth)

    def optimise(self, loss : torch.Tensor):
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def log_loss(self, training_loss, validation_loss, epoch):
        """self.writer.add_scalars('average loss:', 
                                {'training' : training_loss, 
                                 'validation' : validation_loss}, 
                                 epoch)"""
        
        self.writer.add_scalar('average training loss', training_loss, epoch)
        self.writer.add_scalar('average validation loss', validation_loss, epoch)
        
    def log_accuracy(self, training_accuracy, validation_accuracy, epoch):
        """self.writer.add_scalars('average accuracy:', 
                                {'training' : training_accuracy, 
                                 'validation' : validation_accuracy}, 
                                 epoch)"""
        
        self.writer.add_scalar('average training accuracy', training_accuracy, epoch)
        self.writer.add_scalar('average validation accuracy', validation_accuracy, epoch)
        
    def save(self, epoch):
        path = os.path.join('models', self.experiment_name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        torch.save(self.network.state_dict(), os.path.join(path, f'net_{epoch}.pt'))
        torch.save(self.optimiser.state_dict(), os.path.join(path, f'optimiser_{epoch}.pt'))
        if self.scheduler:
            torch.save(self.scheduler.state_dict(), os.path.join(path, f'scheduler_{epoch}.pt'))


class MultiTaskTrainer(Trainer):
    def __init__(self, network : nn.Module, training_datasets : list[Poses3D],
                 validation_datasets : list[Poses3D], optimiser : torch.optim.Adam, 
                 scheduler : torch.optim.lr_scheduler.LinearLR=None,
                 experiment_name='experiment', batch_size=32, epoch=None):
        super().__init__(network, training_datasets,
                         validation_datasets, optimiser,
                         scheduler, experiment_name, 
                         batch_size, epoch)
        self.segmentation_accuracy_function, self.pose_accuracy_function = self.get_accuracy_functions()

    def get_accuracy_functions(self):
        def segmentation_acc(prediction, target):
            batch_size, height, width = target.shape
            prediction = torch.argmax(prediction, 1)
            return torch.sum(prediction == target) / (batch_size * height * width)
        
        def pose_acc(prediction, target): # PDJ 
            prediction = torch.movedim(prediction, 1, 2) 
            target = torch.movedim(target, 1, 2) 

            prediction = reconstruct_skeletons(prediction)
            target = reconstruct_skeletons(target)

            left_shoulder = target[:, 16]
            right_hip = target[:, 2]
            torso_diameter = torch.sqrt(torch.sum(torch.square(left_shoulder - right_hip), axis=1))
            if torso_diameter.dim() == 1:
                torso_diameter = torso_diameter[:, None]
            factor = 0.25
            distances = torch.sqrt(torch.sum(torch.square(prediction - target), axis=2))
            correctly_localised_joints = distances < (factor * torso_diameter)
            batch_size, joints = distances.shape
            return torch.sum(correctly_localised_joints) / (batch_size * joints)
        
        return segmentation_acc, pose_acc
        
    def get_loss_function(self) -> nn.Module:
        class MultiTaskLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.pose_loss = nn.MSELoss(reduction='mean')
                self.segmentation_loss = nn.CrossEntropyLoss(reduction='mean')
            def forward(self, pose_prediction, pose_target, segmentation_prediction, segmentation_target):
                return 0.5 * (self.pose_loss(pose_prediction, pose_target) + self.segmentation_loss(segmentation_prediction, segmentation_target))
            
        return MultiTaskLoss()

    def train(self, epochs):
        for e in range(self.epoch, epochs):
            print(f"Training: \nEpoch {e+1}\n-------------------------------")
            start = time()           
            tr_loss, tr_pose_acc, tr_segmentation_acc = self.train_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average training loss: {tr_loss}, Average training pose accuracy: {tr_pose_acc}, Average training segmentation accuracy: {tr_segmentation_acc} ")
            print(f"Validation: \nEpoch {e+1}\n-------------------------------")
            start = time()       
            val_loss, val_pose_acc, val_segmentation_acc = self.validation_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average validation loss: {val_loss}, Average validation pose accuracy: {val_pose_acc}, Average validation segmentation accuracy: {val_segmentation_acc}")
            self.log_loss(tr_loss, val_loss, e + 1)
            self.log_accuracy(tr_pose_acc, tr_segmentation_acc, val_pose_acc, val_segmentation_acc, e + 1)

            self.save(e + 1)
        self.writer.flush()

        self.writer.close()

    def train_loop(self):
        self.network.train()
        loss_overall = 0
        pose_accuracy_overall, segmentation_accuracy_overall = 0, 0
        samples_count = 0
        for i, dataloader in enumerate(self.training_dataloaders):
            size = len(dataloader.dataset)
            
            for j, sample in enumerate(dataloader):     
                sequences = torch.squeeze(sample['sequences'], 0)
                s, c, h, w = sequences.shape
                poses_groundtruth  = torch.squeeze(sample['key_points'], 0)
                segmentations_groundtruth  = torch.squeeze(sample['segmentations'], 0)
                for k in range(0, s, self.batch_size):
                    sequence = sequences[k:k+self.batch_size]
                    skeletons_groundtruth = poses_groundtruth[k:k+self.batch_size]
                    segmentation_groundtruth = segmentations_groundtruth[k:k+self.batch_size]
                    if len(sequence) < 2:
                        continue
                    samples_count += 1

                    predicted_skeletons, predicted_segmentations = self.network(sequence)
                    loss = self.loss(predicted_skeletons, skeletons_groundtruth, predicted_segmentations, segmentation_groundtruth)
                    self.optimise(loss)
                    
                    skeleton_acc = self.pose_accuracy_function(predicted_skeletons, skeletons_groundtruth)
                    segmentation_acc = self.segmentation_accuracy_function(predicted_segmentations, segmentation_groundtruth)

                    loss_overall += loss.item()
                    pose_accuracy_overall += skeleton_acc.item()
                    segmentation_accuracy_overall += segmentation_acc.item()
                if j % 10 == 0:
                    print(f'dataset: {dataloader.dataset.name}, {j}/{size}, loss: {loss.item()}, pose accuracy: {skeleton_acc.item()}, segmentation accuracy: {segmentation_acc.item()}')

        if self.scheduler:
            self.scheduler.step()
        
        return loss_overall / samples_count, pose_accuracy_overall / samples_count, segmentation_accuracy_overall / samples_count

    def validation_loop(self):
        self.network.eval()
        loss_overall = 0
        pose_accuracy_overall, segmentation_accuracy_overall = 0, 0
        samples_count = 0
        with torch.no_grad():
            for i, dataloader in enumerate(self.validation_dataloaders):
                size = len(dataloader.dataset)
                     
                for j, sample in enumerate(dataloader):     
                    sequences = torch.squeeze(sample['sequences'], 0)
                    s, c, h, w = sequences.shape
                    
                    poses_groundtruth  = torch.squeeze(sample['key_points'], 0)
                    segmentations_groundtruth  = torch.squeeze(sample['segmentations'], 0)
                    for k in range(0, s, self.batch_size):
                        sequence = sequences[k:k+self.batch_size]                    
                        skeletons_groundtruth = poses_groundtruth[k:k+self.batch_size]
                        segmentation_groundtruth = segmentations_groundtruth[k:k+self.batch_size]
                        if len(sequence) < 2:
                            continue
                        samples_count += 1

                        predicted_skeletons, predicted_segmentations = self.network(sequence)
                        loss = self.loss(predicted_skeletons, skeletons_groundtruth, predicted_segmentations, segmentation_groundtruth)
                        
                        skeleton_acc, segmentation_acc = self.accuracy(predicted_skeletons, skeletons_groundtruth, predicted_segmentations, segmentation_groundtruth)

                        loss_overall += loss.item()
                        pose_accuracy_overall += skeleton_acc.item()
                        segmentation_accuracy_overall += segmentation_acc.item()
                    if j % 10 == 0:
                        print(f'dataset: {dataloader.dataset.name}, {j}/{size}, loss: {loss.item()}, pose accuracy: {skeleton_acc.item()}, segmentation accuracy: {segmentation_acc.item()}')

        return loss_overall / samples_count, pose_accuracy_overall / samples_count, segmentation_accuracy_overall / samples_count

    def loss(self, pose_prediction : torch.Tensor, pose_groundtruth : torch.Tensor,
             segmentation_prediction : torch.Tensor, segmentation_groundtruth : torch.Tensor) -> torch.Tensor:
        return self.loss_function(pose_prediction, pose_groundtruth,
                                  segmentation_prediction, segmentation_groundtruth)
    
    def accuracy(self, pose_prediction : torch.Tensor, pose_groundtruth : torch.Tensor,
                 segmentation_prediction : torch.Tensor, segmentation_groundtruth : torch.Tensor) -> tuple[torch.Tensor]:
        return self.pose_accuracy_function(pose_prediction, pose_groundtruth), \
               self.segmentation_accuracy_function(segmentation_prediction, segmentation_groundtruth)
    
    def log_accuracy(self, training_pose_accuracy, training_segmentation_accuracy,
                     validation_pose_accuracy, validation_segmentation_accuracy, epoch):
        self.writer.add_scalar('average training pose accuracy', training_pose_accuracy, epoch)
        self.writer.add_scalar('average validation pose accuracy', validation_pose_accuracy, epoch)
        self.writer.add_scalar('average training segmentation accuracy', training_segmentation_accuracy, epoch)
        self.writer.add_scalar('average validation segmentation accuracy', validation_segmentation_accuracy, epoch)   


class RegressionTrainer(Trainer):
    def __init__(self, network : nn.Module, training_datasets : list[Poses3D],
                 validation_datasets : list[Poses3D], optimiser : torch.optim.Adam, 
                 scheduler : torch.optim.lr_scheduler.LinearLR=None,
                 experiment_name='experiment', batch_size=32, epoch=None):
        super().__init__(network, training_datasets,
                         validation_datasets, optimiser,
                         scheduler, experiment_name, 
                         batch_size, epoch)
    
    def get_accuracy_function(self):   
        def pose_acc(prediction, target): # PDJ factor 0.25
            prediction = torch.movedim(prediction, 1, 2) 
            target = torch.movedim(target, 1, 2) 

            prediction = reconstruct_skeletons(prediction)
            target = reconstruct_skeletons(target)

            left_shoulder = target[:, 16]
            right_hip = target[:, 2]
            torso_diameter = torch.sqrt(torch.sum(torch.square(left_shoulder - right_hip), axis=1))
            if torso_diameter.dim() == 1:
                torso_diameter = torso_diameter[:, None]
            factor = 0.25
            distances = torch.sqrt(torch.sum(torch.square(prediction - target), axis=2))
            correctly_localised_joints = distances < (factor * torso_diameter)
            batch_size, joints = distances.shape
            return torch.sum(correctly_localised_joints) / (batch_size * joints)
        
        return pose_acc
        
    def get_loss_function(self) -> nn.Module:     
        return nn.MSELoss(reduction='mean')


class PoseNetTrainer:
    def __init__(self, network : nn.Module, training_dataloader : DatasetLoader,
                 validation_dataloader: DatasetLoader, optimiser : torch.optim.Adam, 
                 scheduler : torch.optim.lr_scheduler.LinearLR=None,
                 experiment_name='PoseNet_experiment', epoch=None):
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        
        self.experiment_name = experiment_name
        self.joints = network.joints

        self.network = network
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.writer = SummaryWriter(os.path.join('runs', experiment_name))

        if epoch is None:
            self.epoch = 0
        else:
            self.epoch = epoch
            self.load_state()
            self.epoch += 1 # start next epoch

        self.loss_function = self.get_loss_function()
        self.accuracy_function = self.get_accuracy_function()
        self.relative_accuracy_function = self.get_relative_accuracy_function()

    def load_state(self):
        part = self.epoch + 1
        model_path = os.path.join('models', self.experiment_name, f'net_{part}.pt')   
        self.network.load_state_dict(torch.load(model_path))

        optimiser_path = os.path.join('models', self.experiment_name, f'optimiser_{part}.pt')   
        self.optimiser.load_state_dict(torch.load(optimiser_path))

        if self.scheduler:
            scheduler_path = os.path.join('models', self.experiment_name, f'scheduler_{part}.pt')   
            self.scheduler.load_state_dict(torch.load(scheduler_path))

    def get_loss_function(self) -> nn.Module:
        l1_loss = torch.nn.L1Loss()
        def loss(root, joints, target_root, target_joints):
            return l1_loss(root, target_root) + l1_loss(joints, target_joints)
        return loss 
    
    def get_accuracy_function(self) -> nn.Module:
        def acc(root, joints, target_root, target_joints):
            absolute_joints = joints + root[:, :, None]
            absolute_joints_targets = target_joints + target_root[:, :, None]
            joints = torch.cat((root[:,:,None], absolute_joints), 2)
            target_joints = torch.cat((target_root[:,:,None], absolute_joints_targets), 2)
            return torch.mean(torch.sqrt(torch.sum(torch.square(joints - target_joints), axis=1)))            
        return acc
    
    def get_relative_accuracy_function(self) -> nn.Module:
        def acc(relative_joints, relative_target_joints):
            return torch.mean(torch.sqrt(torch.sum(torch.square(relative_joints - relative_target_joints), axis=1)))
             
        return acc

    def train(self, epochs):
        for e in range(self.epoch, epochs):
            print(f"Training: \nEpoch {e+1}\n-------------------------------")
            start = time()           
            tr_loss, tr_acc, tr_rel_acc = self.train_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average training loss: {tr_loss}, Average training accuracy: {tr_acc}")
            print(f"Validation: \nEpoch {e+1}\n-------------------------------")
            start = time()       
            val_loss, val_acc, val_rel_acc = self.validation_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average validation loss: {val_loss}, Average validation accuracy: {val_acc}")
            self.log_loss(tr_loss, val_loss, e + 1)
            self.log_accuracy(tr_acc, val_acc, e + 1)
            self.log_relative_accuracy(tr_rel_acc, val_rel_acc, e + 1)

            self.save(e + 1)
        self.writer.flush()

        self.writer.close()

    def train_loop(self):
        self.network.train()
        loss_overall = 0
        accuracy_overall = 0
        relative_accuracy_overall = 0
        samples_count = 0
        for i, batched_sample in enumerate(self.training_dataloader):
            
            sequences = batched_sample['sequences']
            b, c, h, w = sequences.shape
            skeletons_targets  = batched_sample['key_points']
            skeletons_targets = skeletons_targets[:, :, 1:] # ommit root joint, the relative root joint is always [0, 0, 0]   
            root_targets = batched_sample['root_key_points']

            if b < 2:
                continue
      
            predicted_root_joints, predicted_joints = self.network(sequences)
            loss = self.loss(predicted_root_joints, predicted_joints, root_targets, skeletons_targets)            
            self.optimise(loss)
            acc, rel_acc = self.accuracy(predicted_root_joints, predicted_joints, root_targets, skeletons_targets)

            samples_count += 1
            sample_loss = loss.item()
            sample_acc = acc.item()
            sample_rel_acc = rel_acc.item()

            loss_overall += sample_loss
            accuracy_overall += sample_acc
            relative_accuracy_overall += sample_rel_acc

            if i % 10 == 0:
                name = batched_sample['name']
                print(f'dataset: {name}, batch: {i}, ' +
                        f'loss: {sample_loss:.03f}, ' +
                        f'accuracy: {sample_acc:.03f}, ' +
                        f'relative accuracy: {sample_rel_acc:.03f}')

        if self.scheduler:
            self.scheduler.step()
        
        return loss_overall / samples_count, accuracy_overall / samples_count, relative_accuracy_overall / samples_count

    def validation_loop(self):
        self.network.eval()
        loss_overall = 0
        accuracy_overall = 0
        relative_accuracy_overall = 0
        samples_count = 0

        with torch.no_grad():
            for i, batched_sample in enumerate(self.training_dataloader):              
                sequences = batched_sample['sequences']
                b, c, h, w = sequences.shape
                skeletons_targets  = batched_sample['key_points']
                skeletons_targets = skeletons_targets[:, :, 1:] # ommit root joint, the relative root joint is always [0, 0, 0]   
                root_targets = batched_sample['root_key_points']

                if b < 2:
                    continue
        
                predicted_root_joints, predicted_joints = self.network(sequences)
                loss = self.loss(predicted_root_joints, predicted_joints, root_targets, skeletons_targets)            
                acc, rel_acc = self.accuracy(predicted_root_joints, predicted_joints, root_targets, skeletons_targets)

                samples_count += 1
                sample_loss = loss.item()
                sample_acc = acc.item()
                sample_rel_acc = rel_acc.item()

                loss_overall += sample_loss
                accuracy_overall += sample_acc
                relative_accuracy_overall += sample_rel_acc

                if i % 10 == 0:
                    name = batched_sample['name']
                    print(f'dataset: {name}, batch: {i}, ' +
                            f'loss: {sample_loss:.03f}, ' +
                            f'accuracy: {sample_acc:.03f}, ' +
                            f'relative accuracy: {sample_rel_acc:.03f}')
   
        return loss_overall / samples_count, accuracy_overall / samples_count, relative_accuracy_overall / samples_count

    def loss(self, root_prediction : torch.Tensor, joints_prediction : torch.Tensor, root: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        return self.loss_function(root_prediction, joints_prediction, root, joints)
    
    def accuracy(self, root_prediction : torch.Tensor, joints_prediction : torch.Tensor, root: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        return self.accuracy_function(root_prediction, joints_prediction, root, joints), self.relative_accuracy_function(joints_prediction, joints)

    def log_relative_accuracy(self, training_accuracy, validation_accuracy, epoch): 
        self.writer.add_scalar('average relative training accuracy', training_accuracy, epoch)
        self.writer.add_scalar('average relative validation accuracy', validation_accuracy, epoch)

    def optimise(self, loss : torch.Tensor):
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def log_loss(self, training_loss, validation_loss, epoch):  
        self.writer.add_scalar('average training loss', training_loss, epoch)
        self.writer.add_scalar('average validation loss', validation_loss, epoch)
        
    def log_accuracy(self, training_accuracy, validation_accuracy, epoch):
        self.writer.add_scalar('average training accuracy', training_accuracy, epoch)
        self.writer.add_scalar('average validation accuracy', validation_accuracy, epoch)
        
    def save(self, epoch):
        path = os.path.join('models', self.experiment_name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        torch.save(self.network.state_dict(), os.path.join(path, f'net_{epoch}.pt'))
        torch.save(self.optimiser.state_dict(), os.path.join(path, f'optimiser_{epoch}.pt'))
        if self.scheduler:
            torch.save(self.scheduler.state_dict(), os.path.join(path, f'scheduler_{epoch}.pt'))


class RelativePoseNetTrainer:
    def __init__(self, network : nn.Module, training_dataloader : DatasetLoader,
                 validation_dataloader: DatasetLoader, optimiser : torch.optim.Adam, 
                 scheduler : torch.optim.lr_scheduler.LinearLR=None,
                 experiment_name='PoseNet_experiment', epoch=None, 
                 mean=torch.tensor([28.64750196, 44.02012853, 48.91621137]), 
                 min_=torch.tensor([-5036.994, -5051.3354, -66.55175]),
                 max_=torch.tensor([4979.178, 4967.7715, 2338.0752]),
                 a=torch.tensor(-1), b=torch.tensor(1)):
        
        self.mean = mean
        self.min_ = min_
        self.max_ = max_
        self.a = a
        self.b = b
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        
        self.experiment_name = experiment_name
        self.joints = network.joints

        self.network = network
        self.optimiser = optimiser
        self.scheduler = scheduler
        self.writer = SummaryWriter(os.path.join('runs', experiment_name))

        if epoch is None:
            self.epoch = 0
        else:
            self.epoch = epoch
            self.load_state()
            self.epoch += 1 # start next epoch

        self.loss_function = self.get_loss_function()
        self.accuracy_function = self.get_accuracy_function()

    def load_state(self):
        part = self.epoch + 1
        model_path = os.path.join('models', self.experiment_name, f'net_{part}.pt')   
        self.network.load_state_dict(torch.load(model_path))

        optimiser_path = os.path.join('models', self.experiment_name, f'optimiser_{part}.pt')   
        self.optimiser.load_state_dict(torch.load(optimiser_path))

        if self.scheduler:
            scheduler_path = os.path.join('models', self.experiment_name, f'scheduler_{part}.pt')   
            self.scheduler.load_state_dict(torch.load(scheduler_path))

    def get_loss_function(self) -> nn.Module:
        return torch.nn.L1Loss() 
    
    def get_accuracy_function(self) -> nn.Module:
        def acc(relative_joints, relative_target_joints):
            relative_joints = rescale(relative_joints, self.mean, self.min_, self.max_, self.a, self.b)
            relative_target_joints = rescale(relative_target_joints, self.mean, self.min_, self.max_, self.a, self.b)
            return torch.mean(torch.sqrt(torch.sum(torch.square(relative_joints - relative_target_joints), axis=1)))      
        return acc
    
    def train(self, epochs):
        for e in range(self.epoch, epochs):
            print(f"Training: \nEpoch {e+1}\n-------------------------------")
            start = time()           
            tr_loss, tr_acc = self.train_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average training loss: {tr_loss}, Average training accuracy: {tr_acc}")
            print(f"Validation: \nEpoch {e+1}\n-------------------------------")
            start = time()       
            val_loss, val_acc = self.validation_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average validation loss: {val_loss}, Average validation accuracy: {val_acc}")
            self.log_loss(tr_loss, val_loss, e + 1)
            self.log_accuracy(tr_acc, val_acc, e + 1)

            self.save(e + 1)
        self.writer.flush()

        self.writer.close()

    def train_loop(self):
        self.network.train()
        loss_overall = 0
        accuracy_overall = 0
        samples_count = 0
        for i, batched_sample in enumerate(self.training_dataloader):
            
            sequences = batched_sample['sequences']
            b, c, h, w = sequences.shape
            skeletons_targets  = batched_sample['key_points']
            skeletons_targets = skeletons_targets[:, :, 1:] # ommit root joint, the relative root joint is always [0, 0, 0]   

            if b < 2:
                continue
      
            predicted_joints = self.network(sequences)
            loss = self.loss(predicted_joints, skeletons_targets)            
            self.optimise(loss)
            
            acc = self.accuracy(predicted_joints, skeletons_targets)
            
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
                        f'relative accuracy: {sample_acc:.03f}, ')

        if self.scheduler:
            self.scheduler.step()
        
        return loss_overall / samples_count, accuracy_overall / samples_count

    def validation_loop(self):
        self.network.eval()
        loss_overall = 0
        accuracy_overall = 0
        samples_count = 0

        with torch.no_grad():
            for i, batched_sample in enumerate(self.training_dataloader):              
                sequences = batched_sample['sequences']
                b, c, h, w = sequences.shape
                skeletons_targets  = batched_sample['key_points']
                skeletons_targets = skeletons_targets[:, :, 1:] # ommit root joint, the relative root joint is always [0, 0, 0]   

                if b < 2:
                    continue
        
                predicted_joints = self.network(sequences)
                loss = self.loss(predicted_joints, skeletons_targets)            
                acc = self.accuracy(predicted_joints, skeletons_targets)

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
                            f'relative accuracy: {sample_acc:.03f}, ')
   
        return loss_overall / samples_count, accuracy_overall / samples_count

    def loss(self, joints_prediction : torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        return self.loss_function(joints_prediction, joints)
    
    def accuracy(self, joints_prediction : torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        return self.accuracy_function(joints_prediction, joints)

    def log_accuracy(self, training_accuracy, validation_accuracy, epoch): 
        self.writer.add_scalar('average relative training accuracy', training_accuracy, epoch)
        self.writer.add_scalar('average relative validation accuracy', validation_accuracy, epoch)

    def optimise(self, loss : torch.Tensor):
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def log_loss(self, training_loss, validation_loss, epoch):  
        self.writer.add_scalar('average training loss', training_loss, epoch)
        self.writer.add_scalar('average validation loss', validation_loss, epoch)
                
    def save(self, epoch):
        path = os.path.join('models', self.experiment_name)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        torch.save(self.network.state_dict(), os.path.join(path, f'net_{epoch}.pt'))
        torch.save(self.optimiser.state_dict(), os.path.join(path, f'optimiser_{epoch}.pt'))
        if self.scheduler:
            torch.save(self.scheduler.state_dict(), os.path.join(path, f'scheduler_{epoch}.pt'))

class AbsolutePoseNetTrainer(RelativePoseNetTrainer):
    def __init__(self, network : nn.Module, training_dataloader : DatasetLoader,
                 validation_dataloader: DatasetLoader, optimiser : torch.optim.Adam, 
                 scheduler : torch.optim.lr_scheduler.LinearLR=None,
                 experiment_name='PoseNet_experiment', epoch=None):
        super().__init__(network, training_dataloader, validation_dataloader, optimiser, scheduler, experiment_name, epoch)

    def train_loop(self):
        self.network.train()
        loss_overall = 0
        accuracy_overall = 0
        samples_count = 0
        for i, batched_sample in enumerate(self.training_dataloader):
            
            sequences = batched_sample['sequences']
            b, c, h, w = sequences.shape
            skeletons_targets  = batched_sample['key_points']

            if b < 2:
                continue
      
            predicted_joints = self.network(sequences)
            loss = self.loss(predicted_joints, skeletons_targets)            
            self.optimise(loss)
            
            acc = self.accuracy(predicted_joints, skeletons_targets)
            
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
                        f'absolute accuracy: {sample_acc:.03f}, ')

        if self.scheduler:
            self.scheduler.step()
        
        return loss_overall / samples_count, accuracy_overall / samples_count

    def validation_loop(self):
        self.network.eval()
        loss_overall = 0
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
                loss = self.loss(predicted_joints, skeletons_targets)            
                acc = self.accuracy(predicted_joints, skeletons_targets)

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
                            f'absolute accuracy: {sample_acc:.03f}, ')
   
        return loss_overall / samples_count, accuracy_overall / samples_count

    def log_accuracy(self, training_accuracy, validation_accuracy, epoch): 
        self.writer.add_scalar('average absolute training accuracy', training_accuracy, epoch)
        self.writer.add_scalar('average absolute validation accuracy', validation_accuracy, epoch)

def rescale(pt_cloud, mean, min_, max_, a=-1, b=1):
    pt_cloud = (pt_cloud - a) * (max_[None, :, None] - min_[None, :, None]) / (b - a)
    pt_cloud += min_[None, :, None]
    pt_cloud += mean[None, :, None]
    return pt_cloud