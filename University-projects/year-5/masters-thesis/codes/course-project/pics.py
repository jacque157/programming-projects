from dataset.Dataset import *
from dataset.DataLoader import *
from dataset.Transforms import * 
from visualisation.Visualisation import *
import os
import cv2 

DATASET_PATH = os.path.join('dataset', 'CMU')
SEQUENCE = 1

if __name__ == '__main__':
    CMU_world_pixels = StructuredWorldPointCloudDataset(DATASET_PATH, 'train')

    """skeleton = CMU_world_pixels.load_skeletons(SEQUENCE)[0]
    pt_cloud_0 = CMU_world_pixels.load_structured_point_clouds(SEQUENCE, 1)[0]
    ax = plot_structured_point_cloud(pt_cloud_0)
    plot_skeleton(skeleton, ax)
    plt.show()

    plt.imshow(pt_cloud_0[:, :, 0])
    plt.show()
    plt.imshow(pt_cloud_0[:, :, 1])
    plt.show()
    plt.imshow(pt_cloud_0[:, :, 2])
    plt.show()
    
    pt_cloud_1 = CMU_world_pixels.load_structured_point_clouds(SEQUENCE, 2)[0]
    ax = plot_structured_point_cloud(pt_cloud_1)
    plot_skeleton(skeleton, ax)
    plt.show()
    pt_cloud_2 = CMU_world_pixels.load_structured_point_clouds(SEQUENCE, 3)[0]
    ax = plot_structured_point_cloud(pt_cloud_2)
    plot_skeleton(skeleton, ax)
    plt.show()
    pt_cloud_3 = CMU_world_pixels.load_structured_point_clouds(SEQUENCE, 4)[0]
    ax = plot_structured_point_cloud(pt_cloud_3)
    plot_skeleton(skeleton, ax)
    plt.show()"""

    
    """CMU_camera_pixels = StructuredCameraPointCloudDataset(DATASET_PATH, 'all')
    skeleton = CMU_world_pixels.load_skeletons(SEQUENCE)[0]
    homo_skeleton = np.concatenate((skeleton, np.ones((len(skeleton), 1))), 1)

    data_0 = CMU_camera_pixels[(SEQUENCE - 1) * 4]
    pt_cloud_0 = data_0['point_clouds'][0]
    skeleton = data_0['skeletons'][0]
    homo_skeleton = np.concatenate((skeleton, np.ones((len(skeleton), 1))), 1)
    ax = plot_structured_point_cloud(pt_cloud_0)
    skeleton_0 = (CMU_camera_pixels.extrinsic_matrix(1) @ homo_skeleton.T).T[:, :3]
    plot_skeleton(skeleton_0, ax)
    plt.show()

    plt.imshow(pt_cloud_0[:, :, 0])
    plt.show()
    plt.imshow(pt_cloud_0[:, :, 1])
    plt.show()
    plt.imshow(pt_cloud_0[:, :, 2])
    plt.show()
    plt.imshow(np.abs(pt_cloud_0[:, :, 2]))
    plt.show()
    
    pt_cloud_1 = CMU_camera_pixels[((SEQUENCE - 1) * 4) + 1]['point_clouds'][0]
    ax = plot_structured_point_cloud(pt_cloud_1)
    skeleton_1 = (CMU_camera_pixels.extrinsic_matrix(2) @ homo_skeleton.T).T[:, :3]
    plot_skeleton(skeleton_1, ax)
    plt.show()
    pt_cloud_2 = CMU_camera_pixels[((SEQUENCE - 1) * 4) + 2]['point_clouds'][0]
    ax = plot_structured_point_cloud(pt_cloud_2)
    skeleton_2 = (CMU_camera_pixels.extrinsic_matrix(3) @ homo_skeleton.T).T[:, :3]
    plot_skeleton(skeleton_2, ax)
    plt.show()
    pt_cloud_3 = CMU_camera_pixels[((SEQUENCE - 1) * 4) + 3]['point_clouds'][0]
    ax = plot_structured_point_cloud(pt_cloud_3)
    skeleton_3 = (CMU_camera_pixels.extrinsic_matrix(4) @ homo_skeleton.T).T[:, :3]
    plot_skeleton(skeleton_3, ax)
    plt.show()"""

    """CMU_points = PointCloudDataset(DATASET_PATH, 'all')
    data_0 = CMU_points[0]
    point_cloud = data_0['point_clouds'][0]
    skeleton = data_0['skeletons'][0]
    ax = plot_point_cloud(point_cloud, None, 1)
    plot_skeleton(skeleton, ax)
    plt.show()"""

    CMU_points = PointCloudDataset(DATASET_PATH, 'all')
    min_ = CMU_points.minimum_point()
    max_ = CMU_points.maximum_point()
    avg = CMU_points.average_point()
    transforms = None #Compose([RemoveHandJoints(), AddNormalNoise(5), RandomRotation(30, 30, 180), RandomSampling(1024), ZeroCenter(avg, True), Rescale(min_, max_, -1, 1, True), ToTensor(), ToNumpy()])
    #transforms =  AddNormalNoise(5)
    #transforms =  RandomRotation(30, 30, 180)
    transforms =  RandomSampling(1024)
    dataloader = SequenceDataLoader(CMU_points, 64, transforms, False, False)
    for i, data in enumerate(dataloader):
        point_clouds = data['point_clouds']
        skeletons = data['skeletons']
        for j, (point_cloud, skeleton) in enumerate(zip(point_clouds, skeletons)):
            ax = plot_point_cloud(point_cloud, None, 1)
            plot_skeleton(skeleton, ax)
            plt.show()
            if j >= 10:
                break
        break
    
