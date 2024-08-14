import torch
import os
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from time import time

from DatasetLoader import DatasetLoader

JOINTS = 22

class RelativePoseNetTrainer:
    def __init__(self, network : nn.Module, training_dataloader : DatasetLoader,
                 validation_dataloader: DatasetLoader, optimiser : torch.optim.Adam, 
                 scheduler : torch.optim.lr_scheduler.LinearLR=None,
                 experiment_name='PoseNet_experiment', epoch=None, save_states=True, 
                 mean=torch.tensor([28.64750196, 44.02012853, 48.91621137]), 
                 min_=torch.tensor([-5036.994, -5051.3354, -66.55175]),
                 max_=torch.tensor([4979.178, 4967.7715, 2338.0752]),
                 a=torch.tensor(-1), b=torch.tensor(1)):
        
        self.save_states = save_states
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
        return torch.nn.MSELoss() 
    
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
                            f'relative MPJPE: {sample_acc:.03f}, ')
   
        return loss_overall / samples_count, accuracy_overall / samples_count

    def loss(self, joints_prediction : torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        return self.loss_function(joints_prediction, joints)
    
    def accuracy(self, joints_prediction : torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        return self.accuracy_function(joints_prediction, joints)

    def log_accuracy(self, training_accuracy, validation_accuracy, epoch): 
        self.writer.add_scalar('average relative training MPJPE', training_accuracy, epoch)
        self.writer.add_scalar('average relative validation MPJPE', validation_accuracy, epoch)

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
                 experiment_name='PoseNet_experiment', epoch=None, save_states=True,
                 mean=torch.tensor([28.64750196, 44.02012853, 48.91621137]), 
                 min_=torch.tensor([-5036.994, -5051.3354, -66.55175]),
                 max_=torch.tensor([4979.178, 4967.7715, 2338.0752]),
                 a=torch.tensor(-1), b=torch.tensor(1)):
        super().__init__(network, training_dataloader, validation_dataloader, optimiser, scheduler, experiment_name, epoch, save_states, mean, min_, max_, a, b)

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
                            f'absolute MPJPE: {sample_acc:.03f}, ')
   
        return loss_overall / samples_count, accuracy_overall / samples_count

    def log_accuracy(self, training_accuracy, validation_accuracy, epoch): 
        self.writer.add_scalar('average absolute training MPJPE', training_accuracy, epoch)
        self.writer.add_scalar('average absolute validation MPJPE', validation_accuracy, epoch)

    def get_accuracy_function(self) -> nn.Module:
        def acc(absolute_joints, absolute_target_joints):
            absolute_joints = rescale(absolute_joints, self.mean, self.min_, self.max_, self.a, self.b)
            absolute_target_joints = rescale(absolute_target_joints, self.mean, self.min_, self.max_, self.a, self.b)
            return torch.mean(torch.sqrt(torch.sum(torch.square(absolute_joints - absolute_target_joints), axis=1)))      
        return acc

class PointNetTrainerRelative(RelativePoseNetTrainer):
    def __init__(self, network: nn.Module, training_dataloader: DatasetLoader, validation_dataloader: DatasetLoader, 
                 optimiser: torch.optim.Adam, scheduler: torch.optim.lr_scheduler.LinearLR = None, 
                 experiment_name='PoseNet_experiment', epoch=None, save_states=True, 
                 mean=torch.tensor([28.64750196, 44.02012853, 48.91621137]), 
                 min_=torch.tensor([-5036.994, -5051.3354, -66.55175]), 
                 max_=torch.tensor([4979.178, 4967.7715, 2338.0752]), a=torch.tensor(-1), b=torch.tensor(1)):
        super().__init__(network, training_dataloader, validation_dataloader, optimiser, scheduler, experiment_name, epoch, save_states, mean, min_, max_, a, b)

    def train_loop(self):
        self.network.train()
        loss_overall = 0
        accuracy_overall = 0
        samples_count = 0
        for i, batched_sample in enumerate(self.training_dataloader):
            
            sequences = batched_sample['sequences']
            skeletons_targets  = batched_sample['key_points']
            skeletons_targets = skeletons_targets[:, :, 1:] 
            b, j, n = skeletons_targets.shape
            if b < 2:
                continue
      
            m3x3, m64x64, predicted_joints = self.network(sequences)
            loss = self.loss(m64x64, predicted_joints, skeletons_targets)            
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
            for i, batched_sample in enumerate(self.training_dataloader):    
                          
                sequences = batched_sample['sequences']           
                skeletons_targets  = batched_sample['key_points']
                skeletons_targets = skeletons_targets[:, :, 1:] 
                b, j, n = skeletons_targets.shape

                if b < 2:
                    continue
        
                m3x3, m64x64, predicted_joints = self.network(sequences)
                loss = self.loss(m64x64, predicted_joints, skeletons_targets)            
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
                            f'absolute MPJPE: {sample_acc:.03f}, ')
   
        return loss_overall / samples_count, accuracy_overall / samples_count

    def log_accuracy(self, training_accuracy, validation_accuracy, epoch): 
        self.writer.add_scalar('average absolute training MPJPE', training_accuracy, epoch)
        self.writer.add_scalar('average absolute validation MPJPE', validation_accuracy, epoch)

    def get_accuracy_function(self) -> nn.Module:
        def acc(absolute_joints, absolute_target_joints):
            absolute_joints = rescale(absolute_joints, self.mean, self.min_, self.max_, self.a, self.b)
            absolute_target_joints = rescale(absolute_target_joints, self.mean, self.min_, self.max_, self.a, self.b)
            return torch.mean(torch.sqrt(torch.sum(torch.square(absolute_joints - absolute_target_joints), axis=1)))      
        return acc
    
    def get_loss_function(self)-> nn.Module:
        l2_loss = torch.nn.MSELoss()
        weight = 0.001
        def pointNet_loss(trans_matrix, prediction, ground_truth):
            device = prediction.device
            identity = torch.eye(trans_matrix.size(1)).to(device)
            reg_term = torch.norm(identity - torch.bmm(trans_matrix, trans_matrix.transpose(2, 1)))
            loss = l2_loss(prediction, ground_truth) + reg_term * weight
            return loss
        return pointNet_loss 
    
    def loss(self, matrix, joints_prediction : torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
        return self.loss_function(matrix, joints_prediction, joints)

def rescale(pt_cloud, mean, min_, max_, a=-1, b=1):
    pt_cloud = (pt_cloud - a) * (max_[None, :, None] - min_[None, :, None]) / (b - a)
    pt_cloud += min_[None, :, None]
    pt_cloud += mean[None, :, None]
    return pt_cloud