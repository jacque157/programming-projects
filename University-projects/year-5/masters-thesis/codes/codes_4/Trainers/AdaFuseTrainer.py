import torch
import os
from torch import nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from time import time
from Trainers.CenterNetTrainer import Trainer as CenterNetTrainer
from Trainers.CenterNetTrainer import MPJPE
from Networks.AdaFuseRep.core.evaluate import accuracy

def frozen_backbone_bn(model, backbone_name='resnet'):
    for name, m in model.named_modules():
        if backbone_name in name:
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # logger.info(name)
                m.eval()
            else:
                pass

class Trainer(CenterNetTrainer):
    def __init__(self, network, training_dataloader,
                 validation_dataloader, optimiser, 
                 scheduler=None, experiment_name='AdaFuse',
                 epoch=None, save_states=True):
        super().__init__(None, network, training_dataloader,
                 validation_dataloader, optimiser, 
                 scheduler, experiment_name,
                 epoch, save_states)

    def train(self, epochs):
        for e in range(self.epoch, epochs):
            print(f"Training: \nEpoch {e+1}\n-------------------------------")
            start = time()           
            tr_loss, tr_acc, _ = self.train_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average training loss: {tr_loss}, Average training accuracy: {tr_acc}")
            print(f"Validation: \nEpoch {e+1}\n-------------------------------")
            start = time()       
            val_loss, val_acc, mpjpe = self.validation_loop()
            elapsed_time = time() - start
            print(f"Time taken: {elapsed_time:0.4f} seconds")
            print(f"Average validation loss: {val_loss}, Average validation accuracy: {val_acc}, Average validation MPJPE: {mpjpe}")
            self.log_loss(tr_loss, val_loss, e + 1)
            self.log_accuracy(tr_acc, val_acc, e + 1)
            self.log_mpjpe(mpjpe, val_acc, e + 1)

            if (self.save_states):
                self.save(e + 1)
        self.writer.flush()

        self.writer.close()
        
    def train_loop(self):
        self.network.train()
        frozen_backbone_bn(self.network.network)
        loss_overall = 0
        accuracy_overall = 0
        samples_count = 0
        for i, batched_sample in enumerate(self.training_dataloader):
            heat_maps_targets = batched_sample['heat_maps']
            b, v, c, h, w = heat_maps_targets.shape
            heat_maps_targets = heat_maps_targets.reshape(b * v, c, h, w)
            skeletons_targets = batched_sample['skeletons']      
            b, j, n = skeletons_targets.shape
            if b * v < 2:
                continue
            heat_maps, loss, _ = self.network(batched_sample, train=True)
            if len(heat_maps.shape) == 4:
                heat_maps = heat_maps[None]
            heat_maps = heat_maps.reshape(b * v, c, h, w)
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
        
        return loss_overall / samples_count, accuracy_overall / samples_count, None

    def validation_loop(self):
        self.network.eval()
        loss_overall = 0
        accuracy_overall = 0
        mpjpe_overall = 0
        samples_count = 0

        with torch.no_grad():
            for i, batched_sample in enumerate(self.validation_dataloader):                           
                heat_maps_targets = batched_sample['heat_maps']
                b, v, c, h, w = heat_maps_targets.shape
                heat_maps_targets = heat_maps_targets.reshape(b * v, c, h, w)
                skeletons_targets = batched_sample['skeletons']      
                b, j, n = skeletons_targets.shape
                if b * v < 2:
                    continue
                heat_maps, loss, _ = self.network(batched_sample, train=False)
                if len(heat_maps.shape) == 4:
                    heat_maps = heat_maps[None]
                heat_maps = heat_maps.reshape(b * v, c, h, w)

                if torch.any(loss.isnan()):
                    print('nan occured in loss')
                    continue
                
                _, acc, _, _ = accuracy(heat_maps.detach().cpu().numpy(),
                                        heat_maps_targets.detach().cpu().numpy(),
                                        thr=0.083)
                mpjpe = MPJPE(predicted_3D_skeleton, skeletons_targets)
                samples_count += 1
                sample_loss = loss.item()
                sample_acc = acc.item()
                sample_mpjpe = mpjpe.item()
                torch.cuda.empty_cache()
                
                loss_overall += sample_loss
                accuracy_overall += sample_acc
                mpjpe_overall += sample_mpjpe

                if i % 20 == 0:
                    name = batched_sample['name']
                    print(f'dataset: {name}, batch: {i}, ' +
                            f'loss: {sample_loss:.03f}, ' +
                            f'average accuracy: {sample_acc:.03f}, ' +
                          f'average MPJPE: {sample_mpjpe}')
   
        return loss_overall / samples_count, accuracy_overall / samples_count, mpjpe_overall / samples_count
    
    def log_loss(self, training_loss, validation_loss, epoch):  
        self.writer.add_scalar('average 2d training loss', training_loss, epoch)
        self.writer.add_scalar('average 2d validation loss', validation_loss, epoch)

    def log_mpjpe(self, training_mpjpe, validation_mpjpe, epoch):
        super().log_accuracy(training_mpjpe, validation_mpjpe, epoch)
        
    def log_accuracy(self, training_accuracy, validation_accuracy, epoch): 
        self.writer.add_scalar('average training 2D accuracy', training_accuracy, epoch)
        self.writer.add_scalar('average validation 2D accuracy', validation_accuracy, epoch)
