import os
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
from data_loader import KinopticDataset
from model import RandLANet
from transforms import ToTensor
import numpy as np
import torch
from torch.utils.data import sampler
from torch.utils import data
from torch import optim
from torch import nn
from datetime import datetime


class Trainer:
    def __init__(self, batch_size=10, training_p=0.7, validation_p=0.2, seed=42, path='models'):
        # testing_p = 1 - (training_p + validation_p)

        self.dataset = KinopticDataset("../../../data", ToTensor())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"training on device: {self.device}")

        number_of_joints = 15
        self.randla_net = RandLANet(3, number_of_joints, device=self.device)
        self.randla_net.apply(self.weights_init)
        self.data_path = path
        self.loss_function = nn.CrossEntropyLoss()

        # self.batch_size = batch_size

        training_indexes, validation_indexes, testing_indexes = self.get_indexes(training_p, validation_p, seed)

        train_sampler = sampler.SubsetRandomSampler(training_indexes)
        validation_sampler = sampler.SubsetRandomSampler(validation_indexes)
        testing_sampler = sampler.SubsetRandomSampler(testing_indexes)

        self.training_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        self.validation_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=validation_sampler, num_workers=4)
        #self.testing_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=testing_sampler)

        self.optimiser = optim.Adam(self.randla_net.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimiser, step_size=20, gamma=0.5)

        self.training_loss_list = []
        self.validation_loss_list = []

    def train(self, epochs, validate=False, start=0):
        now = datetime.now()
        val_writer = SummaryWriter(f'runs/experiment/validation{now.strftime("%Y%m%d-%H%M%S")}')
        tr_writer = SummaryWriter(f'runs/experiment/training{now.strftime("%Y%m%d-%H%M%S")}')
        acc_val_writer = SummaryWriter(f'runs/experiment/validation_accuracy{now.strftime("%Y%m%d-%H%M%S")}')
        acc_tr_writer = SummaryWriter(f'runs/experiment/training_accuracy{now.strftime("%Y%m%d-%H%M%S")}')

        for epoch in range(start, epochs):
            time0 = time.time()
            self.randla_net.train()
            # self.loss_function.train()
            running_loss = 0
            running_accuracy = 0
            for batch_index, cloud in enumerate(self.training_loader):
                point_cloud, annotation = self.get_data_and_ground_truth(cloud)
                point_cloud = point_cloud.permute(0, 2, 1)

                prediction = self.randla_net(point_cloud)
                loss = self.loss_function(prediction, annotation)

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()
                running_accuracy += Trainer.calculate_accuracy(prediction, annotation)

                if batch_index % 20 == 0:
                    print('-' * 40)
                    print(f"Epoch: {epoch}, batch: {batch_index}, loss: {loss}")

            training_loss = running_loss / len(self.training_loader)
            accuracy = running_accuracy / len(self.training_loader)
            tr_writer.add_scalar('training loss', training_loss, epoch)
            acc_tr_writer.add_scalar('training accuracy', accuracy, epoch)
            self.training_loss_list.append(training_loss)

            self.scheduler.step()
            # torch.cuda.empty_cache()

            if validate:
                validation_loss, accuracy = self.validate_training(epoch)
                val_writer.add_scalar('validation loss', validation_loss, epoch)
                acc_val_writer.add_scalar('validation accuracy', accuracy, epoch)
                self.validation_loss_list.append(validation_loss)
                fig = self.plot_losses(self.training_loss_list, self.validation_loss_list, epoch)
                tr_writer.add_figure('Training vs Validation Loss', fig, global_step=epoch)
                # torch.cuda.empty_cache()

            self.save_state(self.data_path, epoch)

            print(f"\nTime elapsed from last epoch (in minutes) = {(time.time() - time0) / 60}")

    def resume(self, path, start, epochs, validate):
        self.load_state(path, start)
        self.train(epochs, validate, start)

    def save_state(self, path, epoch):
        torch.save(self.randla_net.state_dict(), f"{path}/model_{epoch}.pth")
        torch.save(self.optimiser.state_dict(), f"{path}/optimiser_{epoch}.pth")
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler_{epoch}.pth")
        Trainer.store_losses(self.training_loss_list, self.validation_loss_list, f"{path}/losses.txt")

    def load_state(self, path, epoch):
        self.randla_net.load_state_dict(torch.load(f"{path}/model_{epoch}.pth"))
        self.optimiser.load_state_dict(torch.load(f"{path}/optimiser_{epoch}.pth"))
        self.scheduler.load_state_dict(torch.load(f"{path}/scheduler_{epoch}.pth"))
        self.training_loss_list, self.validation_loss_list = Trainer.load_losses(f"{path}/losses")

    @staticmethod
    def plot_losses(tr_loss_list, val_loss_list, e):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.plot(list(range(e + 1)), tr_loss_list, '-o', label='training loss')
        plt.plot(list(range(e + 1)), val_loss_list, '-o', label='validation loss')
        plt.legend(loc='upper right')
        return fig

    def get_data_and_ground_truth(self, cloud):
        point_cloud = cloud['point_cloud'].to(device=self.device, dtype=torch.float)
        annotation = cloud['annotations'].to(device=self.device, dtype=torch.long)
        return point_cloud, annotation

    def validate_training(self, epoch):
        running_loss = 0
        accuracy = 0
        self.randla_net.eval()
        # self.loss_function.eval()
        for batch_index, cloud in enumerate(self.validation_loader):
            point_cloud, annotation = self.get_data_and_ground_truth(cloud)
            point_cloud = point_cloud.permute(0, 2, 1)
            prediction = self.randla_net(point_cloud)
            loss = self.loss_function(prediction, annotation)
            running_loss += loss.item()
            accuracy += Trainer.calculate_accuracy(prediction, annotation)
            if batch_index % 20 == 0:
                print('-' * 40)
                print(f"Epoch: {epoch}, batch: {batch_index}, validation loss: {loss}")

        return running_loss / len(self.validation_loader), accuracy / len(self.validation_loader)

    @staticmethod
    def calculate_accuracy(prediction, truth):
        prediction = prediction.max(1)[1]
        batch_accuracy = torch.div(torch.sum((truth == prediction), axis=1), truth.size(1))
        return torch.div(torch.sum(batch_accuracy), truth.size(0)).item()

    @staticmethod
    def store_indexes(training_indexes, validation_indexes, testing_indexes, file):
        with open(file, 'w') as f:
            print("Training indexes:", file=f)
            print(training_indexes, file=f)
            print("Validation indexes:", file=f)
            print(validation_indexes, file=f)
            print("Testing_indexes:", file=f)
            print(testing_indexes, file=f)

    @staticmethod
    def load_indexes(file):
        with open(file) as f:
            f.readline()
            training_indexes = list(map(int, f.readline()[1:-2].split(',')))
            f.readline()
            validation_indexes = list(map(int, f.readline()[1:-2].split(',')))
            f.readline()
            testing_indexes = list(map(int, f.readline()[1:-2].split(',')))
        return training_indexes, validation_indexes, testing_indexes

    @staticmethod
    def store_losses(training_losses, validation_losses, file):
        with open(file, 'w') as f:
            print("Training loss:", file=f)
            print(training_losses, file=f)
            print("Validation loss:", file=f)
            print(validation_losses, file=f)

    @staticmethod
    def load_losses(file):
        training_losses = [], validation_losses = []
        with open(file) as f:
            f.readline()
            training_losses = list(map(int, f.readline()[1:-2].split(',')))
            f.readline()
            validation_losses = list(map(int, f.readline()[1:-2].split(',')))
        return training_losses, validation_losses

    @staticmethod
    def indexes_exist(file):
        return os.path.exists(file)

    def create_indexes(self, training_p, validation_p, seed):
        indexes = list(range(len(self.dataset)))
        np.random.RandomState(seed).shuffle(indexes)

        training_split = int(np.floor(len(self.dataset) * training_p))
        validation_split = training_split + int(np.floor(len(self.dataset) * validation_p))

        training_indexes = indexes[:training_split]
        validation_indexes = indexes[training_split:validation_split]
        testing_indexes = indexes[validation_split:]

        return training_indexes, validation_indexes, testing_indexes

    def get_indexes(self, training_p, validation_p, seed):
        if Trainer.indexes_exist("indexes.txt"):
            training_indexes, validation_indexes, testing_indexes = Trainer.load_indexes("indexes.txt")
        else:
            training_indexes, validation_indexes, testing_indexes = self.create_indexes(training_p, validation_p, seed)
            Trainer.store_indexes(training_indexes, validation_indexes, testing_indexes, "indexes.txt")

        return training_indexes, validation_indexes, testing_indexes

    def weights_init(self, w):
        classname = w.__class__.__name__

        if classname.find('Conv2d') != -1:
            if w.weight is not None:
                torch.nn.init.xavier_normal_(w.weight.data)
            if w.bias is not None:
                torch.nn.init.constant_(w.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            if w.weight is not None:
                torch.nn.init.xavier_normal_(w.weight.data)
            if w.bias is not None:
                torch.nn.init.constant_(w.bias.data, 0.0)


# %load_ext tensorboard


# %tensorboard --logdir runs


if __name__ == '__main__':
    tr = Trainer(2)
    """for batch_index, cloud in enumerate(tr.training_loader):
        point_cloud, annotation = tr.get_data_and_ground_truth(cloud)
        cloud, pose = point_cloud[0].transpose(1, 0), annotation[0]
        print(cloud.shape)
        print(pose.shape)
        Sketch.plot_annotated_ptcloud(cloud.numpy(), pose.numpy())"""

    print(tr.calculate_accuracy(torch.rand(32, 15, 2048), torch.randint(0, 15, (32, 2048))))
        
    tr.train(2, True)
