import os
import torch

from dataset.Dataset import StructuredCameraPointCloudDataset
from dataset.DataLoader import SequenceDataLoader
from dataset.Transforms import *
from Networks.CenterNet import CenterNetDLA
from Trainers.CenterNetTrainer import MPJPE, predict_3D_skeleton, rescale

from visualisation import Visualisation
import matplotlib.pyplot as plt

DATASET_PATH = os.path.join('dataset', 'CMU')
DEVICE = 'cuda:0'
EXPERIMENT_NAME = 'CenterNetDLA34PoseEstimation'
BATCH_SIZE = 16
REDUCTION = 4
BEST_MODEL = 23#15

def compute_per_sample_per_joint_MPJPE(dataset, network):
    def dist(a, b):
        return torch.sum((a - b) ** 2, 1)  ** 0.5

    result = []
    batch_size = 32
    for sequence, sample in enumerate(dataset):
        frame = 0
        point_clouds = sample['point_clouds']
        for i in range(0, (len(point_clouds) // batch_size) + 1, batch_size):
            point_clouds_batched = point_clouds[i : i + batch_size].to(DEVICE)
            if len(point_clouds_batched) == 0:
                break
            heat_maps, offsets, depths = network(point_clouds_batched)
            predicted_3D_skeleton = predict_3D_skeleton(heat_maps, offsets, depths,
                                                        sample['intrinsic_matrix_inverted'].to(DEVICE),
                                                        sample['rotation_matrix_inverted'].to(DEVICE),
                                                        sample['translation_vector_inverted'].to(DEVICE), REDUCTION)
            gt_skeletons = sample['skeletons'][i : i + batch_size].to(DEVICE)
            distances = dist(predicted_3D_skeleton, gt_skeletons)
            for j, distance in enumerate(distances):
                result.append({'sequence' : sequence, 'frame' : frame + j, 'MPJPEs' : [error.item() for error in distance]})
            frame += len(point_clouds_batched)
        if sequence % 20 == 0:
            print(f'Sequence: {sequence} / {len(dataset)}')
    return result

def compute_error(dataloader, network):
    samples = 0
    error_sum = 0
    for i, sample in enumerate(dataloader):
        point_clouds = sample['point_clouds']
        b, c, h, w = point_clouds.shape
        if b == 0:
            continue
        samples += b
        heat_maps, offsets, depths = network(point_clouds)
        predicted_3D_skeleton = predict_3D_skeleton(heat_maps, offsets, depths,
                                                    sample['intrinsic_matrix_inverted'],
                                                    sample['rotation_matrix_inverted'],
                                                    sample['translation_vector_inverted'], REDUCTION)

        for skeleton_predicted, skeleton_target in zip(predicted_3D_skeleton.detach().cpu(), sample['skeletons'].detach().cpu()):
            skeleton_numpy = torch.moveaxis(skeleton_target, 0, 1).numpy()
            ax = Visualisation.plot_skeleton(skeleton_numpy)
            ax.set_title('Ground Truth')
            plt.show()

            skeleton_numpy = torch.moveaxis(skeleton_predicted, 0, 1).numpy()
            ax = Visualisation.plot_skeleton(skeleton_numpy)
            ax.set_title('Prediction')
            plt.show()

            
        error = MPJPE(predicted_3D_skeleton, sample['skeletons'])
        print(f'Batch {i}, MPJPE: {error}')
        error_sum += error * b  
    mean_error = error_sum / samples
    return mean_error

def compute_per_sample_MPJPE(dataset, network):
    result = []
    batch_size = 64
    for sequence, sample in enumerate(dataset):
        frame = 0
        point_clouds = sample['point_clouds']
        for i in range(0, (len(point_clouds) // batch_size) + 1, batch_size):
            point_clouds_batched = point_clouds[i : i + batch_size].to(DEVICE)
            if len(point_clouds_batched) == 0:
                break
            heat_maps, offsets, depths = network(point_clouds_batched)
            predicted_3D_skeleton = predict_3D_skeleton(heat_maps, offsets, depths,
                                                        sample['intrinsic_matrix_inverted'].to(DEVICE),
                                                        sample['rotation_matrix_inverted'].to(DEVICE),
                                                        sample['translation_vector_inverted'].to(DEVICE), REDUCTION)

            for j, (skeleton_predicted, skeleton_target) in enumerate(zip(predicted_3D_skeleton.detach().cpu(), sample['skeletons'][i : i + batch_size])):
                error = MPJPE(skeleton_predicted[None], skeleton_target[None])
                result.append({'sequence' : sequence, 'frame' : frame + j, 'MPJPE' : error.item()})
            frame += len(point_clouds_batched)
        if sequence % 20 == 0:
            print(f'Sequence: {sequence} / {len(dataset)}')
    return result

def order_per_sample_MPJPE(per_sample_MPJPE):
    per_sample_MPJPE.sort(key=lambda x : x['MPJPE'])

def structured2unstructured(structured_point_cloud, eps=1e-4):
    xs, ys, zs  = structured_point_cloud[:, :, 0], structured_point_cloud[:, :, 1], structured_point_cloud[:, :, 2]
    xs_mask = np.logical_and(-eps < xs, xs < eps )
    ys_mask = np.logical_and(-eps < ys, ys < eps )
    zs_mask = np.logical_and(-eps < zs, zs < eps )
    mask = np.logical_not(np.logical_and(np.logical_and(xs_mask, ys_mask), zs_mask))
    return structured_point_cloud[mask]

def farthest_point_sampling(points, n):
    # https://minibatchai.com/2021/08/07/FPS.html
    points_left = np.arange(len(points)) 
    sample_inds = np.zeros(n, dtype='int') 
    dists = np.ones_like(points_left) * float('inf') 
    selected = 0
    sample_inds[0] = points_left[selected]
    points_left = np.delete(points_left, selected) 

    for i in range(1, n):
        last_added = sample_inds[i-1]
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) 
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) 
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]
        points_left = np.delete(points_left, selected)

    return points[sample_inds]

def plot_selected_skeletons(dataset, network, indices, n=2048):
    for sequence, frame in indices:
        sample = dataset[sequence]
        point_cloud = sample['point_clouds'][frame].to(DEVICE)
        heat_map, offset, depth = network(point_cloud[None])
        predicted_skeleton = predict_3D_skeleton(heat_map, offset, depth,
                                                    sample['intrinsic_matrix_inverted'].to(DEVICE),
                                                    sample['rotation_matrix_inverted'].to(DEVICE),
                                                    sample['translation_vector_inverted'].to(DEVICE), REDUCTION)
        target_skeleton = np.moveaxis(sample['skeletons'][frame].numpy(), 0, 1)
        predicted_skeleton = np.moveaxis(predicted_skeleton[0].detach().cpu().numpy(), 0, 1)

        if n >= 0:
            dense_point_cloud = None
            index_start = sequence - (sequence % 4)
            for camera_index in range(4):
                index = index_start + camera_index
                sample = dataset[index]
                point_clouds_structured = sample['point_clouds']
                point_clouds_structured_scaled_back = rescale(point_clouds_structured, sample['center'], sample['min_'], sample['max_'], sample['a'], sample['b'])
                point_clouds_structured_scaled_back = np.moveaxis(point_clouds_structured_scaled_back[frame].numpy(), 0, -1)
                point_cloud_unstructured = structured2unstructured(point_clouds_structured_scaled_back)
                point_cloud_world = point_cloud_unstructured  + sample['translation_vector_inverted'][None, :].numpy()
                point_cloud_world = (sample['rotation_matrix_inverted'].numpy() @ point_cloud_world.T).T
                if dense_point_cloud is None:
                    dense_point_cloud = point_cloud_world
                else:
                    dense_point_cloud = np.append(dense_point_cloud, point_cloud_world, 0)
            if n < len(dense_point_cloud):
                point_cloud = farthest_point_sampling(dense_point_cloud, n)
            else:
                point_cloud = dense_point_cloud
        else:
            point_clouds_structured_scaled_back = rescale(point_cloud[None].detach().cpu(), sample['center'], sample['min_'], sample['max_'], sample['a'], sample['b'])
            point_clouds_structured_scaled_back = np.moveaxis(point_clouds_structured_scaled_back[0].numpy(), 0, -1)
            point_cloud_unstructured = structured2unstructured(point_clouds_structured_scaled_back)
            point_cloud_world = point_cloud_unstructured  + sample['translation_vector_inverted'][None, :].numpy()
            point_cloud_world = (sample['rotation_matrix_inverted'].numpy() @ point_cloud_world.T).T
            point_cloud = point_cloud_world
        Visualisation.plot_example(point_cloud, predicted_skeleton, target_skeleton)

def top_mid_worst_k_predictions(per_sample_MPJPE, k=5):
    order_per_sample_MPJPE(per_sample_MPJPE)
    per_sample_MPJPE_list = [(sample['sequence'], sample['frame'], sample['MPJPE']) for sample in per_sample_MPJPE]
    start = per_sample_MPJPE_list[0 : k]
    end = per_sample_MPJPE_list[len(per_sample_MPJPE_list) - k:]
    mid_start = (len(per_sample_MPJPE_list) // 2) - (k // 2)
    mid = per_sample_MPJPE_list[mid_start : mid_start + k]

    result = []
    result.extend(start)
    result.extend(mid)
    result.extend(end)
    return result
    

def generate_per_sample_MPJPE_file(test_dataset_transformed, network, output_path='per_sample_MPJPE.txt'):
    if os.path.exists(output_path):
        per_sample_MPJPE = []
        with open(output_path) as file:
            for line in file:
                values = line.split()
                seq, frame, MPJPE = int(values[0]), int(values[1]), float(values[2])
                per_sample_MPJPE.append({'sequence' : seq, 'frame' : frame, 'MPJPE' : MPJPE})
    else:
        per_sample_MPJPE = compute_per_sample_MPJPE(test_dataset_transformed, network)
        with open(output_path, 'w') as file:
            for sample in per_sample_MPJPE:
                seq, frame, MPJPE = sample['sequence'], sample['frame'], sample['MPJPE']
                print(seq, frame, MPJPE, file=file)
    return per_sample_MPJPE

def generate_per_sample_per_joint_MPJPE_file(test_dataset_transformed, network, output_path='per_sample_per_joint_MPJPE.txt'):
    if os.path.exists(output_path):
        per_sample_MPJPEs = []
        with open(output_path) as file:
            for line in file:
                values = line.split()
                seq, frame, MPJPEs = int(values[0]), int(values[1]), [float(num) for num in (values[2:])]
                per_sample_MPJPEs.append({'sequence' : seq, 'frame' : frame, 'MPJPEs' : MPJPEs})
    else:
        per_sample_MPJPEs = compute_per_sample_per_joint_MPJPE(test_dataset_transformed, network)
        with open(output_path, 'w') as file:
            for sample in per_sample_MPJPEs:
                seq, frame, MPJPEs = sample['sequence'], sample['frame'], sample['MPJPEs']
                print(seq, frame, *MPJPEs, file=file)
    return per_sample_MPJPEs

"""def plot_per_sample_MPJPE(per_sample_MPJPE, splits=100, min_=None, max_=None, parts=None):
    MPJPE_list = [sample['MPJPE'] for sample in per_sample_MPJPE]
    print(np.mean(MPJPE_list))
    Visualisation.plot_MPJPEs(MPJPE_list, splits, min_, max_)
    Visualisation.plot_MPJPEs_histogram(MPJPE_list, splits)
    if parts is None:
        parts = 5
    Visualisation.plot_MPJPEs_histogram_2(MPJPE_list, parts, min_, max_)"""

def compute_plot_MPJPE(test_dataset_transformed, network):
    per_sample_MPJPE = generate_per_sample_MPJPE_file(test_dataset_transformed, network, output_path='per_sample_MPJPE.txt')
    MPJPE_list = [sample['MPJPE'] for sample in per_sample_MPJPE]
    print(np.mean(MPJPE_list))
    Visualisation.plot_MPJPEs(MPJPE_list, 1000, None, None)
    Visualisation.plot_MPJPEs_histogram(MPJPE_list, 50)
    Visualisation.plot_MPJPEs_histogram_2(MPJPE_list, 5, min_=None, max_=200)

def compute_per_joint_MPJPE(test_dataset_transformed, network):
    MPJPE_per_joint = generate_per_sample_per_joint_MPJPE_file(test_dataset_transformed, network, output_path='per_sample_per_joint_MPJPE.txt')
    MPJPE_per_joint_list = np.array([sample['MPJPEs'] for sample in MPJPE_per_joint])
    joints_errors = np.mean(MPJPE_per_joint_list, 0)
    stds = np.std(MPJPE_per_joint_list, 0)
    for i, (error, std) in enumerate(zip(joints_errors, stds)):
        print(f"joint: {i}, MPJPE: {error}, std: {std}")

    Visualisation.plot_per_joint_error(MPJPE_per_joint_list)

def plot_best_wors_predictions(test_dataset_transformed, network):
    per_sample_MPJPE = generate_per_sample_MPJPE_file(test_dataset_transformed, network, output_path='per_sample_MPJPE.txt')
    selected_per_sample_MPJPE = top_mid_worst_k_predictions(per_sample_MPJPE, k=3)
    indices = []
    for seq, frame, mpjpe in selected_per_sample_MPJPE:
            #seq, frame, mpjpe = result['sequence'], result['frame'], result['MPJPE']
        indices.append((seq, frame))
        print(f'sequence: {seq}, frame: {frame}, MPJPE: {mpjpe}')
    plot_selected_skeletons(test_dataset_transformed, network, indices, n=8192)
         
if __name__ == '__main__':
    test_dataset = StructuredCameraPointCloudDataset(DATASET_PATH, 'test')
    min_ = test_dataset.min(centered=True)
    max_ = test_dataset.max(centered=True)
    avg = test_dataset.avg()
    transforms = Compose([RemoveHandJoints(),
                          ZeroCenter(avg, transform_skeletons=False),
                          Rescale(min_, max_, -1, 1, transform_skeletons=False),
                          ToTensor()])
    test_dataloader = SequenceDataLoader(test_dataset, BATCH_SIZE,
                                          transforms,
                                          shuffle_sequences=True,
                                          shuffle_frames=True,
                                          device=DEVICE)
    test_dataset_transformed = StructuredCameraPointCloudDataset(DATASET_PATH, 'test', transforms=transforms)
    
    network = CenterNetDLA(n_channels=3, n_joints=22)
    network = network.to(DEVICE)
    path = os.path.join('models', f'{EXPERIMENT_NAME}', f'net_{BEST_MODEL}.pt')
    network.load_state_dict(torch.load(path))
    network.eval()

    with torch.no_grad():
        compute_plot_MPJPE(test_dataset_transformed, network)

        compute_per_joint_MPJPE(test_dataset_transformed, network)

        plot_best_wors_predictions(test_dataset_transformed, network)
        
        #error = compute_error(test_dataloader, network)
        #print(f'MPJPE: {error}')
        

