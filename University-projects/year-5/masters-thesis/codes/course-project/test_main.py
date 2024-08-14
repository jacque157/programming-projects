from dataset.Dataset import *
from dataset.DataLoader import *
from dataset.Transforms import * 
from visualisation.Visualisation import *
import os
import cv2 

DATASET_PATH = os.path.join('dataset', 'CMU')

if __name__ == '__main__':
    CMU_world_pixels = StructuredWorldPointCloudDataset(DATASET_PATH, 'train')
    min_ = CMU_world_pixels.minimum_world_pixel()
    max_ = CMU_world_pixels.maximum_world_pixel()
    avg = CMU_world_pixels.average_pixel()
    transforms = None #Compose([RemoveHandJoints(), ZeroCenter(avg, True), Rescale(min_, max_, -1, 1, True), ToTensor(), ToNumpy()])
    dataloader = SequenceDataLoader(CMU_world_pixels, 64, transforms, False, False)
    for i, data in enumerate(dataloader):
        point_clouds = data['point_clouds']
        #print(point_clouds.shape)
        skeletons = data['skeletons']
        for j, (point_cloud, skeleton) in enumerate(zip(point_clouds, skeletons)):
            ax = plot_structured_point_cloud(point_cloud)
            plot_skeleton(skeleton, ax)
            plt.show()
            if j >= 10:
                break
        break

    CMU_camera_pixels = StructuredCameraPointCloudDataset(DATASET_PATH, 'train')
    min_ = CMU_camera_pixels.minimum_camera_pixel()
    max_ = CMU_camera_pixels.maximum_camera_pixel()
    avg = CMU_camera_pixels.average_camera_point()
    transforms = None #Compose([RemoveHandJoints(), ToTensor(), ToNumpy()]) #Compose([ZeroCenter(avg), Rescale(min_, max_), ToTensor(), ToNumpy()])
    dataloader = SequenceDataLoader(CMU_camera_pixels, 64, transforms, False, False)
    for i, data in enumerate(dataloader):
        point_clouds = data['point_clouds']
        skeletons = data['skeletons']
        skeletons_2D = data['skeletons_2D']
        skeleton_depths = data['skeleton_depths']
        heat_maps = data['heat_maps']
        for j, (point_cloud, skeleton, skeleton_2D, heat_map) in enumerate(zip(point_clouds, skeletons, skeletons_2D, heat_maps)):
            ax = plot_structured_point_cloud(point_cloud)
            plt.show()
            h, w, c = point_cloud.shape

            x = point_cloud[:, :, 0]
            y = point_cloud[:, :, 1]
            z = point_cloud[:, :, 2]
            mask =  np.logical_not(np.logical_and(np.logical_and(x == 0, y == 0), z == 0))
            points = point_cloud[mask]
            points = np.concatenate((points, np.ones((len(points), 1))), 1)
            points_2D = (CMU_camera_pixels.intrinsic_matrix() @ points.T).T
            depths = points_2D[:, -1, None]
            points_2D /= depths
            points_2D[:, -1] = depths.flatten()
            
            depth_map = np.zeros((h, w))
            for x, y, d in points_2D:
                x = int(np.round(x))
                y = int(np.round(y))
                depth_map[y, x] = d
                    
            plot_depth_map(point_cloud[:, :, -1])
            plt.show()
            plot_depth_map(depth_map)
            plt.show()
            summed_heatmaps = np.sum(heat_map, 2)
            plt.imshow(summed_heatmaps)
            plt.show()

            img = depth_map#np.zeros(point_cloud.shape)
            img /= img.max()
            img *= 255
            img = np.ascontiguousarray(img, dtype=np.uint8)
            #img /= img.max()
            #img *= 255
            for (x, y) in skeleton_2D:
                img = cv2.circle(img, (int(x), int(y)), 4, 255, 2)
            plt.imshow(img)
            plt.show()
            if j >= 10:
                break
        break
    
    CMU_points = PointCloudDataset(DATASET_PATH, 'train')
    min_ = CMU_points.minimum_point()
    max_ = CMU_points.maximum_point()
    avg = CMU_points.average_point()
    transforms = None #Compose([RemoveHandJoints(), AddNormalNoise(5), RandomRotation(30, 30, 180), RandomSampling(1024), ZeroCenter(avg, True), Rescale(min_, max_, -1, 1, True), ToTensor(), ToNumpy()])
    dataloader = SequenceDataLoader(CMU_points, 64, transforms, False, False)
    for i, data in enumerate(dataloader):
        point_clouds = data['point_clouds']
        print(point_clouds.shape)
        skeletons = data['skeletons']
        for j, (point_cloud, skeleton) in enumerate(zip(point_clouds, skeletons)):
            ax = plot_point_cloud(point_cloud, None, 1)
            plot_skeleton(skeleton, ax)
            plt.show()
            if j >= 10:
                break
        break

