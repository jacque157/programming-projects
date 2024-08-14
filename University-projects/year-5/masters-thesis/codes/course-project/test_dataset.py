from dataset.Dataset import *
from dataset.DataLoader import *
from dataset.Transforms import * 
from visualisation.Visualisation import *
import os
import cv2 

DATASET_PATH = os.path.join('dataset', 'CMU')

"""def predict_3D_skeleton(heat_maps, offsets, depths,
                        intrinsic_matrix_inverted,
                        rotation_matrix_inverted,
                        translation_vector_inverted,
                        stride=4):
    
        b, j, h, w = heat_maps.shape
        heat_maps_flattened = torch.flatten(heat_maps, 2)
        idx = torch.argmax(heat_maps_flattened, 2)[:, :, None]
        xs = idx % w
        ys = idx // w
        positions_2D = torch.stack((xs, ys), 1).squeeze()

        offsets_flattened = torch.flatten(offsets, 2)
        x_offs = offsets_flattened[:, 0, None, :].expand(-1, j, -1)
        y_offs = offsets_flattened[:, 1, None, :].expand(-1, j, -1)
        relevant_x_offs = torch.take_along_dim(x_offs, idx, 2).squeeze()
        relevant_y_offs = torch.take_along_dim(y_offs, idx, 2).squeeze()
        relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 1)

        depths_flattened = -torch.flatten(depths, 2)
        relevant_depths = torch.take_along_dim(depths_flattened, idx, 2).squeeze()[:, None, :]

        positions_corrected = (positions_2D + relevant_offsets) * stride
        points = torch.concatenate((positions_corrected, torch.ones(b, 1, j)), 1)
        points *= relevant_depths
        points_3D = torch.bmm(rotation_matrix_inverted.expand(b, 3, 3),
                              translation_vector_inverted[None, :, None] + \
                              torch.bmm(intrinsic_matrix_inverted.expand(b, 3, 3), points))
        return points_3D"""

def MPJPE(prediction, ground_truth):
    return torch.mean(torch.sqrt(torch.sum((prediction - ground_truth) ** 2, 1)))

def predict_3D_skeleton(heat_maps, offsets, depths,
                        intrinsic_matrix_inverted,
                        rotation_matrix_inverted,
                        translation_vector_inverted,
                        stride=4):
    
        b, j, h, w = heat_maps.shape
        heat_maps_flattened = torch.flatten(heat_maps, 2)
        idx = torch.argmax(heat_maps_flattened, 2)[:, :, None]
        xs = idx % w
        ys = idx // w
        positions_2D = torch.stack((xs, ys), 1).squeeze()

        """offsets_flattened = torch.flatten(offsets, 2)
        x_offs = offsets_flattened[:, 0, None, :].expand(-1, j, -1)
        y_offs = offsets_flattened[:, 1, None, :].expand(-1, j, -1)
        relevant_x_offs = torch.take_along_dim(x_offs, idx, 2).squeeze()
        relevant_y_offs = torch.take_along_dim(y_offs, idx, 2).squeeze()
        relevant_offsets = torch.stack((relevant_x_offs, relevant_y_offs), 1)

        depths_flattened = -torch.flatten(depths, 2)
        relevant_depths = torch.take_along_dim(depths_flattened, idx, 2).squeeze()[:, None, :]"""

        positions_corrected = (positions_2D + offsets) * stride
        points = torch.concatenate((positions_corrected, torch.ones(b, 1, j, device=positions_corrected.device)), 1)
        points *= -depths
        """points_3D = torch.bmm(rotation_matrix_inverted.expand(b, 3, 3),
                              translation_vector_inverted[None, :, None] + \
                              torch.bmm(intrinsic_matrix_inverted.expand(b, 3, 3), points))"""
        points_flat = torch.reshape(torch.moveaxis(points, 1, 0), (3, b * j))
        points_3D_flat = torch.matmul(rotation_matrix_inverted, torch.matmul(intrinsic_matrix_inverted, points_flat) + translation_vector_inverted[:, None])
        points_3D = torch.moveaxis(torch.reshape(points_3D_flat, (3, b, j)), 0, 1)
        return points_3D

if __name__ == '__main__':
    CMU_camera_pixels = StructuredCameraPointCloudDataset(DATASET_PATH, 'train')
    min_ = CMU_camera_pixels.minimum_camera_pixel()
    max_ = CMU_camera_pixels.maximum_camera_pixel()
    avg = CMU_camera_pixels.average_camera_point()
    transforms = Compose([RemoveHandJoints(), ZeroCenter(avg, transform_skeletons=False), Rescale(min_, max_, transform_skeletons=False), ToTensor()])
    dataloader = SequenceDataLoader(CMU_camera_pixels, 64, transforms, True, True, 'cuda:0')

    for i, data in enumerate(dataloader):#enumerate(CMU_camera_pixels):#enumerate(dataloader):
        #data = transforms(data)
        #print(i)
        point_clouds = data['point_clouds']
        if len(point_clouds) == 0:
            print('oops')
        skeletons = data['skeletons']
        skeletons_2D = data['skeletons_2D']
        skeleton_depths = data['skeletons_depths']
        heat_maps = data['heat_maps']
        K_ = data['intrinsic_matrix_inverted']
        R_ = data['rotation_matrix_inverted']
        t_ = data['translation_vector_inverted']
        offsets = (skeletons_2D / 4) - torch.round(skeletons_2D / 4)
        #offsets = 0
        predicted_skeletons = predict_3D_skeleton(heat_maps, offsets, skeleton_depths,
                                                 K_, R_, t_, 4)
        print(MPJPE(predicted_skeletons, skeletons))
        
        """for j, (d, skely_2d, skeleton_pr, skeleton_gt) in enumerate(zip(np.moveaxis(skeleton_depths.numpy(), 1, 2), np.moveaxis(skeletons_2D.numpy(), 1, 2), np.moveaxis(predicted_skeletons.numpy(), 1, 2), np.moveaxis(skeletons.numpy(), 1, 2))):
            metric = np.mean(np.sqrt(np.sum((skeleton_pr - skeleton_gt) ** 2, 1)))
            if metric > 1.0 or np.isnan(metric):
                print(i, j)
                print(metric)
                print(skeleton_gt)
                print(skeleton_pr)
                ax = plot_skeleton(skeleton_gt)
                plt.show()
                ax = plot_skeleton(skeleton_pr)
                plt.show()

                points_2D = np.concatenate((skely_2d, np.ones((len(skely_2d), 1))), 1) * -d
                points_3D = (R_.numpy() @ ((K_.numpy() @ points_2D.T) + t_.numpy()[:, None])).T
                
                ax = plot_skeleton(points_3D)
                plt.show()"""
                
        
    """for i, data in enumerate(dataloader):
        point_clouds = data['point_clouds']
        skeletons = data['skeletons']
        skeletons_offsets = data['2D_skeletons_offsets']
        skeleton_depths = data['2D_skeletons_depths']
        heat_maps = data['heat_maps']
        K_ = data['intrinsic_matrix_inverted']
        R_ = data['rotation_matrix_inverted']
        t_ = data['translation_vector_inverted']
        prediction = predict_3D_skeleton(heat_maps, skeletons_offsets, skeleton_depths, K_, R_, t_, 4)
        print(MPJPE(prediction, skeletons))
        print(skeleton_depths.shape)
        for j, (skeleton, gt_skeleton) in enumerate(zip(np.moveaxis(prediction.numpy(), 1, 2), np.moveaxis(skeletons.numpy(), 1, 2))):
            metric = np.mean(np.sqrt(np.sum((skeleton - gt_skeleton) ** 2, 1)))
            if metric > 0.1:
                print(j)
                #print(metric)
                #print(gt_skeleton)
                #print(skeleton)
                d = np.moveaxis(skeleton_depths.numpy(), 1, 3)[j]
                plt.imshow(d)
                plt.show()
                ax = plot_skeleton(gt_skeleton)
                plt.show()
                ax = plot_skeleton(skeleton)
                plt.show()
                
        
    CMU_camera_pixels = StructuredCameraPointCloudDataset(DATASET_PATH, 'train')
    min_ = CMU_camera_pixels.minimum_camera_pixel()
    max_ = CMU_camera_pixels.maximum_camera_pixel()
    avg = CMU_camera_pixels.average_camera_point()
    transforms = Compose([RemoveHandJoints(), ToTensor(), ToNumpy()]) #Compose([ZeroCenter(avg), Rescale(min_, max_), ToTensor(), ToNumpy()])
    dataloader = SequenceDataLoader(CMU_camera_pixels, 64, transforms, False, False)
    for i, data in enumerate(dataloader):
        point_clouds = data['point_clouds']
        skeletons = data['skeletons']

        skeletons_offsets = data['2D_skeletons_offsets']
        skeleton_depths = data['2D_skeletons_depths']
        heat_maps = data['heat_maps']
        for j, (point_cloud, skeleton,
                offsets, depths, heat_map) in enumerate(zip(point_clouds, skeletons,
                                                            skeletons_offsets, skeleton_depths, heat_maps)):
            ax = plot_structured_point_cloud(point_cloud)
            plt.show()
            print(heat_map.shape)
            skeleton_2D = []
            h, w, j = heat_map.shape
            heat_map_flat = np.reshape(heat_map, (h * w, j))
            idx_1D = np.argmax(heat_map_flat, 0)
            xs = idx_1D % w
            ys = idx_1D // w
            skeletons_2D = np.dstack((xs, ys)).squeeze()

            ws = depths[ys, xs]
            x_offsets = offsets[ys, xs, 0]
            y_offsets = offsets[ys, xs, 1]

            relevant_offsets = np.dstack((x_offsets, y_offsets)).squeeze()
            print(skeletons_2D)
            print(ws)
            print(relevant_offsets)
            print()

            skeletons_2D_corrected = (skeletons_2D + relevant_offsets) * 4
            print(skeletons_2D_corrected)

            points = np.concatenate((skeletons_2D_corrected, np.ones((len(skeletons_2D_corrected), 1))), 1)
            points = points * ws[:, None]

            K_ = data['intrinsic_matrix_inverted']
            R_ = data['rotation_matrix_inverted']
            t_ = data['translation_vector_inverted']
            
            points_3D = (R_ @ ((K_ @ points.T).T  + t_).T).T
            print(points_3D.shape)
            
            print(points_3D)
            print(skeleton)

            ax = plot_skeleton(points_3D)
            plt.show()"""

"""plt.imshow(offsets[:, :, 0])
            plt.show()
            plt.imshow(offsets[:, :, 1])
            plt.show()
            plt.imshow(depths[:, :])
            plt.show()
            for k in range(len(skeletons)): 
                joint_heat_map = heat_map[:, :, k]
                h, w = joint_heat_map.shape
                idx_1D = np.argmax(joint_heat_map)
                y = idx_1D // w
                x = idx_1D % w
                plt.imshow(joint_heat_map)
                print(x, y)
                plt.show()
                
                #skeleton_2D.append((x, y, 1))
            #skeleton_2D = np.array(skeleton_2D)

            
            
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
        break"""
    

