import torch
import torch.nn as  nn

import Dataset
import os 
from Transforms import ToTensor, ToDevice
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from Networks.PoseNet import PoseResNet18NoBatchNorm


DATASET_PATH = os.path.join('..', 'codes', 'Dataset')

def get_frame(dataset):
    sample = dataset[0]
    return sample['sequences'][0][None], sample['key_points'][0][None]

def get_scaled_frame(dataset):
    sample = dataset[0]
    pt_cloud = sample['sequences'][0]
    skels = sample['key_points'][0]
    mean = torch.mean(pt_cloud.view(3, -1), 1)
    pt_cloud -= mean[:, None, None]
    skels -= mean[:, None]
    min_ = torch.min(pt_cloud.view(3, -1), 1)[0]
    max_ = torch.max(pt_cloud.view(3, -1), 1)[0]

    pt_cloud = -1 + ((pt_cloud - min_[:, None, None]) * (1 - -1) / (max_[:, None, None] - min_[:, None, None]))
    skels = -1 + ((skels - min_[:, None]) * (1 - -1) / (max_[:, None] - min_[:, None]))
    
    return pt_cloud[None], skels[None]

def rescale_frame(dataset, predicted):
    predicted = predicted.view(3, 22)
    sample = dataset[0]
    pt_cloud = sample['sequences'][0]
    mean = torch.mean(pt_cloud.view(3, -1), 1)
    pt_cloud -= mean[:, None, None]
    
    min_ = torch.min(pt_cloud.view(3, -1), 1)[0]
    max_ = torch.max(pt_cloud.view(3, -1), 1)[0]
    
    predicted = (predicted - -1) * (max_[:, None] - min_[:, None]) / (1 - -1)
    predicted += min_[:, None]
    predicted += mean[:, None]
    
    return predicted

def build_model():
    return nn.Sequential(nn.Conv2d(3, 128, 3, 1, 0),
                         nn.AvgPool2d(2, 2),                 
                         nn.LeakyReLU(),
                         
                         nn.AdaptiveAvgPool2d(1), 
                         nn.Flatten(),
                         nn.Linear(128, 64),
                         nn.LeakyReLU(),
                         
                         nn.Linear(64, 22 * 3))

def build_model():
    return PoseResNet18NoBatchNorm(3, 22)

def acc_fun(pred, truth):
    return torch.mean(torch.sqrt(torch.sum((pred - truth) ** 2, 1)))

print(torch.cuda.is_available())
transform = transforms.Compose((ToTensor(),
                               ToDevice('cuda:0')))

dataset = Dataset.Poses3D(DATASET_PATH, 'CMU', transform)

model = build_model()
model = model.to(device='cuda:0')
loss_fun = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
experiment_name = "PoseResNet18-no_batch_norm_normalised_avg_pool_leaky_ReLU_no_dropout"
writer = SummaryWriter(os.path.join('runs', experiment_name))

for e in range(1000):
    pt_cloud, target = get_scaled_frame(dataset)
    pt_cloud = pt_cloud.to(device='cuda:0')
    target = target.to(device='cuda:0')
    prediction = torch.reshape(model(pt_cloud), (3, 22))[None]
    loss = loss_fun(prediction, target)   

    
    prediction = rescale_frame(dataset, prediction.detach().cpu())
    target = rescale_frame(dataset, target.detach().cpu())
    acc = acc_fun(prediction, target)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    writer.add_scalar('Single Frame Loss', loss.item(), e)
    writer.add_scalar('Single Frame MPJPE', acc.item(), e)
    if e % 50 == 0:
        print(f'epoch: {e} loss: {loss.item()} acc: {acc.item()}')

