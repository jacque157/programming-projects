from torch import nn

from data_loader import KinopticDataset
from model import RandLANet
from transforms import ToTensor
import torch
from torch.utils.data import sampler
from torch.utils import data
from visualisation import Sketch


class Tester:
    def __init__(self, model_path, batch_size=10):
        # testing_p = 1 - (training_p + validation_p)

        self.dataset = KinopticDataset("../../data", ToTensor())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"testing on device: {self.device}")

        number_of_joints = 15
        self.randla_net = RandLANet(3, number_of_joints, device=self.device)
        self.randla_net.load_state_dict(torch.load(model_path, self.device))
        self.randla_net.eval()

        self.loss_function = nn.CrossEntropyLoss()
        # self.batch_size = batch_size

        testing_indexes = self.get_indexes()
        testing_sampler = sampler.RandomSampler(testing_indexes)

        self.testing_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=testing_sampler)

    def test(self):
        running_accuracy = 0
        running_loss = 0
        running_miou = 0

        for batch_index, cloud in enumerate(self.testing_loader):
            point_cloud, annotation = self.get_data_and_ground_truth(cloud)
            point_cloud = point_cloud.permute(0, 2, 1)
            prediction = self.randla_net(point_cloud)
            loss = self.loss_function(prediction, annotation)

            running_loss += loss.item()

            batch_accuracy = torch.mean(Tester.calculate_accuracy(prediction, annotation))
            miou = Tester.mIoU(prediction, annotation)
            running_accuracy += batch_accuracy.item()
            running_miou += miou

            if batch_index % 20 == 0:
                print('-' * 40)
                print(f"Batch: {batch_index}, loss: {loss}, accuracy: {batch_accuracy.item() * 100}%, mIoU : {miou}")

                cloud = point_cloud[0] #.transpose(1, 0)
                pose_p = prediction.max(1)[1][0]
                pose_t = annotation[0]
                fig_p = Sketch.plot_annotated_ptcloud(cloud.detach().cpu().numpy(), pose_p.detach().cpu().numpy())
                fig_gt = Sketch.plot_annotated_ptcloud(cloud.detach().cpu().numpy(), pose_t.detach().cpu().numpy())
                Sketch.plot_difference(cloud.detach().cpu().numpy(), pose_t.detach().cpu().numpy(), pose_p.detach().cpu().numpy())

        testing_loss = running_loss / len(self.testing_loader)
        print(f"\nParameters: {Tester.count_parameters(self.randla_net)}")
        print(f"\nTesting loss: {testing_loss}")
        print(f"\nAccuracy: {running_accuracy / len(self.testing_loader)}")
        print(f"\nmIoU: {running_miou / len(self.testing_loader)}")

    def generate_images(self, acc=0.9):
        print(f"\nParameters: {Tester.count_parameters(self.randla_net)}")
        for batch_index, cloud in enumerate(self.testing_loader):
            point_cloud_b, annotation_b = self.get_data_and_ground_truth(cloud)
            point_cloud_b = point_cloud_b.permute(0, 2, 1)
            prediction_b = self.randla_net(point_cloud_b)
            key_points = 15
            for i in range(len(point_cloud_b)):
                point_cloud, annotation, prediction = point_cloud_b[i], annotation_b[i], prediction_b[i]
                cloud = point_cloud

                #point_cloud = point_cloud.permute(0, 2, 1)
                pose_p = prediction.max(0)[1]
                pose_t = annotation

                accuracy = torch.div(torch.sum((pose_t == pose_p)), pose_t.size(0)).item()
                if accuracy >= acc:
                    continue

                print(i)
                print(f'Accuracy: {accuracy * 100}%')

                iou = 0
                for key in range(key_points):
                    intersection = torch.logical_and(pose_p == key, pose_t == key).sum()
                    union = torch.logical_or(pose_p == key, pose_t == key).sum()
                    if union == 0:
                        union = 1
                    iou += (intersection / union).item()
                miou = iou / key_points

                print(f'mIoU: {miou * 100}%')

                if self.device == 'cuda':
                    Sketch.plot_annotated_ptcloud(cloud.detach().cpu().numpy(), pose_p.detach().cpu().numpy())
                    Sketch.plot_annotated_ptcloud(cloud.detach().cpu().numpy(), pose_t.detach().cpu().numpy())
                    Sketch.plot_difference(cloud.detach().cpu().numpy(), pose_t.detach().cpu().numpy(),
                                           pose_p.detach().cpu().numpy())
                else:
                    Sketch.plot_annotated_ptcloud(cloud.detach().numpy(), pose_p.detach().numpy())
                    Sketch.plot_annotated_ptcloud(cloud.detach().numpy(), pose_t.detach().numpy())
                    Sketch.plot_difference(cloud.detach().numpy(), pose_t.detach().numpy(), pose_p.detach().numpy())

    @staticmethod
    def calculate_accuracy(prediction, truth):
        prediction = prediction.max(1)[1]
        return torch.div(torch.sum((truth == prediction), axis=1), truth.size(1))

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_data_and_ground_truth(self, cloud):
        point_cloud = cloud['point_cloud'].to(device=self.device, dtype=torch.float)
        annotation = cloud['annotations'].to(device=self.device, dtype=torch.long)
        return point_cloud, annotation

    @staticmethod
    def load_indexes(file):
        with open(file) as f:
            f.readline()
            training_indexes = list(map(int, f.readline()[1:-2].split(',')))
            f.readline()
            validation_indexes = list(map(int, f.readline()[1:-2].split(',')))
            f.readline()
            testing_indexes = list(map(int, f.readline()[1:-2].split(',')))
        return testing_indexes

    def get_indexes(self):
        return Tester.load_indexes("indexes.txt")

    @staticmethod
    def mIoU(prediction, ground_truth, key_points=15):
        prediction = prediction.max(1)[1]
        iou = 0
        for key in range(key_points):
            intersection = torch.logical_and(prediction == key, ground_truth == key).sum((0, 1))
            union = torch.logical_or(prediction == key, ground_truth == key).sum((0, 1))
            if union == 0:
                union = 1
            iou += (intersection / union).item()
        return iou / key_points


# %load_ext tensorboard


# %tensorboard --logdir runs


if __name__ == '__main__':
    ts = Tester('models/randla/model_64.pth', 16)
    ts.test()
    #ts.generate_images()




