import torch
import os
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from time import time
from Trainers.CenterNetTrainer import Trainer as CenterNetTrainer
from Networks.AdaFuseRep.core.evaluate import accuracy

class Trainer(CenterNetTrainer):
    def __init__(self, network, training_dataloader,
                 validation_dataloader, optimiser, 
                 scheduler=None, experiment_name='PoseResNetBaseline',
                 epoch=None, save_states=True):
        super().__init__(None, network, training_dataloader,
                 validation_dataloader, optimiser, 
                 scheduler, experiment_name,
                 epoch, save_states)

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
            skeletons_targets = batched_sample['skeletons']      
            b, j, n = skeletons_targets.shape
            if b < 2:
                continue
      
            heat_maps, features = self.network(sequences)
            loss = self.network.loss(heat_maps, heat_maps_targets)
            if torch.any(loss.isnan()):
                print('nan occured in loss')
                continue
            self.optimise(loss)
            
            _, acc, _, _ = accuracy(heat_maps.detach().cpu().numpy(),
                                        heat_maps_targets.detach().cpu().numpy(),
                                        thr=0.083)
            
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
                        f'average accuracy: {sample_acc:.03f}, ')

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
                skeletons_targets = batched_sample['skeletons']   
                b, j, n = skeletons_targets.shape
                if b < 2:
                    continue
          
                heat_maps, features = self.network(sequences)
                loss = self.network.loss(heat_maps, heat_maps_targets)

                if torch.any(loss.isnan()):
                    print('nan occured in loss')
                    continue
                
                _, acc, _, _ = accuracy(heat_maps.detach().cpu().numpy(),
                                        heat_maps_targets.detach().cpu().numpy(),
                                        thr=0.083)
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
                            f'average accuracy: {sample_acc:.03f}, ')
   
        return loss_overall / samples_count, accuracy_overall / samples_count
          
    def log_loss(self, training_loss, validation_loss, epoch):  
        self.writer.add_scalar('average 2d training loss', training_loss, epoch)
        self.writer.add_scalar('average 2d validation loss', validation_loss, epoch)
        
    def log_accuracy(self, training_accuracy, validation_accuracy, epoch): 
        self.writer.add_scalar('average training 2D accuracy', training_accuracy, epoch)
        self.writer.add_scalar('average validation 2D accuracy', validation_accuracy, epoch)
