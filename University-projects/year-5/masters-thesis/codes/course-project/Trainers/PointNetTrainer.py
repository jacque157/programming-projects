import torch
import os
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from time import time


class Trainer:
    def __init__(self, network, training_dataloader,
                 validation_dataloader, optimiser, 
                 scheduler=None, experiment_name='PointNetRegression',
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
            sequences = batched_sample['point_clouds']
            skeletons_targets  = batched_sample['skeletons']
            b, j, n = skeletons_targets.shape
            if b < 2:
                continue
      
            m3x3, m64x64, predicted_joints = self.network(sequences)
            loss = self.network.loss(predicted_joints, skeletons_targets, m64x64)            
            self.optimise(loss)
            
            acc = self.MPJPE(predicted_joints, batched_sample)
            
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
                sequences = batched_sample['point_clouds']
                skeletons_targets  = batched_sample['skeletons']
                b, j, n = skeletons_targets.shape
                if b < 2:
                    continue
        
                m3x3, m64x64, predicted_joints = self.network(sequences)
                loss = self.network.loss(predicted_joints, skeletons_targets, m64x64)            
                acc = self.MPJPE(predicted_joints, batched_sample)

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
    
    def MPJPE(self, predictions, data):
        if 'center' in data:
            mean = data['center']
        else:
            device = a.device
            mean = torch.zeros(3, device=device)
                
        if 'a' in data:
            a = data['a']
            b = data['b']
            min_ = data['min_']
            max_ = data['max_']
            predictions_scaled = rescale(predictions, mean, min_, max_, a, b)
            skeletons_scaled = rescale(data['skeletons'], mean, min_, max_, a, b)
        else:
            predictions_scaled = predictions + mean[None, :, None]
            skeletons_scaled = data['skeletons'] + mean[None, :, None]
        return torch.mean(torch.sqrt(torch.sum(torch.square(predictions_scaled - skeletons_scaled), axis=1)))

    def log_loss(self, training_loss, validation_loss, epoch):  
        self.writer.add_scalar('average training loss', training_loss, epoch)
        self.writer.add_scalar('average validation loss', validation_loss, epoch)
        
    def log_accuracy(self, training_accuracy, validation_accuracy, epoch): 
        self.writer.add_scalar('average training MPJPE', training_accuracy, epoch)
        self.writer.add_scalar('average validation MPJPE', validation_accuracy, epoch)

    def optimise(self, loss : torch.Tensor):
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
    pt_cloud = (pt_cloud - a) * (max_[None, :, None] - min_[None, :, None]) / (b - a)
    pt_cloud += min_[None, :, None]
    pt_cloud += mean[None, :, None]
    return pt_cloud
