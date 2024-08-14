from data_loader import KinopticDataset
from loss import PointNetLoss
from point_net import PointNetSegmentation
from point_net import PointNetPartSegmentation
from transforms import ToTensor
import torch
from torch.utils.data import sampler
from torch.utils import data
from visualisation import Sketch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class Tester:
    def __init__(self, model_path, batch_size=10):
        # testing_p = 1 - (training_p + validation_p)

        self.dataset = KinopticDataset("../data", ToTensor())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"testing on device: {self.device}")

        number_of_joints = 15
        self.pointNet = PointNetSegmentation(self.device, number_of_joints).to(self.device)
        self.pointNet.load_state_dict(torch.load(model_path, self.device))
        self.pointNet.eval()

        self.loss_function = PointNetLoss(self.device)
        # self.batch_size = batch_size

        testing_indexes = self.get_indexes()
        testing_sampler = sampler.RandomSampler(testing_indexes)

        self.testing_loader = data.DataLoader(self.dataset, batch_size=batch_size, sampler=testing_sampler)

    def test(self):
        running_accuracy = 0
        running_loss = 0
        running_miou = 0
        now = datetime.now()
        test_t_writer = SummaryWriter(f'runs/experiment/testing_ground_truth{now.strftime("%Y%m%d-%H%M%S")}')
        test_p_writer = SummaryWriter(f'runs/experiment/testing_prediction{now.strftime("%Y%m%d-%H%M%S")}')

        for batch_index, cloud in enumerate(self.testing_loader):
            point_cloud, annotation = self.get_data_and_ground_truth(cloud)
            mat3x3, mat64x64, prediction = self.pointNet(point_cloud)
            loss = self.loss_function(mat64x64, prediction, annotation)

            running_loss += loss.item()

            batch_accuracy = torch.mean(Tester.calculate_accuracy(prediction, annotation))
            miou = Tester.mIoU(prediction, annotation)
            running_accuracy += batch_accuracy.item()
            running_miou += miou

            if batch_index % 20 == 0:
                print('-' * 40)
                print(f"Batch: {batch_index}, loss: {loss}, accuracy: {batch_accuracy.item() * 100}%, mIoU : {miou}")

                cloud = point_cloud[0].transpose(1, 0)
                pose_p = prediction.max(1)[1][0]
                pose_t = annotation[0]
                fig_p = Sketch.plot_annotated_ptcloud(cloud.detach().cpu().numpy(), pose_p.detach().cpu().numpy(),
                                                      False)
                fig_gt = Sketch.plot_annotated_ptcloud(cloud.detach().cpu().numpy(), pose_t.detach().cpu().numpy(),
                                                       False)

                test_t_writer.add_figure('Ground truth', fig_gt, global_step=batch_index)
                test_p_writer.add_figure('Prediction', fig_p, global_step=batch_index)

        testing_loss = running_loss / len(self.testing_loader)
        print(f"\nTesting loss: {testing_loss}")
        print(f"\nAccuracy: {running_accuracy / len(self.testing_loader)}")
        print(f"\nmIoU: {running_miou / len(self.testing_loader)}")

    def generate_images(self):
        for batch_index, cloud in enumerate(self.testing_loader):
            point_cloud_b, annotation_b = self.get_data_and_ground_truth(cloud)
            mat3x3, mat64x64, prediction_b = self.pointNet(point_cloud_b)
            key_points = 15
            for i in range(len(point_cloud_b)):
                point_cloud, annotation, prediction = point_cloud_b[i], annotation_b[i], prediction_b[i]
            
                cloud = point_cloud.transpose(1, 0)
                pose_p = prediction.max(0)[1]
                pose_t = annotation
                if self.device == 'cuda':
                    Sketch.plot_annotated_ptcloud(cloud.detach().cpu().numpy(), pose_p.detach().cpu().numpy())
                    Sketch.plot_annotated_ptcloud(cloud.detach().cpu().numpy(), pose_t.detach().cpu().numpy())
                else:
                    Sketch.plot_annotated_ptcloud(cloud.detach().numpy(), pose_p.detach().numpy())
                    Sketch.plot_annotated_ptcloud(cloud.detach().numpy(), pose_t.detach().numpy())

                iou = 0
                for key in range(key_points):
                    intersection = torch.logical_and(pose_p == key, pose_t == key).sum()
                    union = torch.logical_or(pose_p == key, pose_t == key).sum()
                    if union == 0:
                        union = 1
                    iou += (intersection / union).item()
                miou = iou / key_points
                accuracy = torch.div(torch.sum((pose_t == pose_p)), pose_t.size(0)).item()
                
                print(f'Accuracy: {accuracy * 100}%')
                print(f'mIoU: {miou * 100}%')

    @staticmethod
    def calculate_accuracy(prediction, truth):
        prediction = prediction.max(1)[1]
        return torch.div(torch.sum((truth == prediction), axis=1), truth.size(1))

    """"@staticmethod
    def plot_losses(tr_loss_list, val_loss_list, e):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        plt.plot(list(range(e + 1)), tr_loss_list, '-o', label='training loss')
        plt.plot(list(range(e + 1)), val_loss_list, '-o', label='validation loss')
        plt.legend(loc='upper right')
        return fig"""

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
    ts = Tester('models/PNSemSeg.pth', 32)
    #ts.test()
    ts.generate_images()
