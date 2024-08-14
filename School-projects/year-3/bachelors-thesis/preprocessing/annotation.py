import numpy as np
from tools import Tools
from visualisation import Sketch

class Annotation:

    @staticmethod
    def annotate_ptcloud(pt_cloud, joints):
        distances = np.sqrt(np.sum((joints - pt_cloud[:, None]) ** 2, -1))
        annotation = np.argmin(distances, -1)
        return annotation

    @staticmethod
    def annotate(pt_cloud_paths, joints_paths, result_paths, start=0):
        error_logs = []
        for i in range(start, len(pt_cloud_paths)):
            try:
                print(f"{(i / len(pt_cloud_paths)) * 100:2f}% completion")
                pt_cloud_path, joints_path = pt_cloud_paths[i], joints_paths[i]
                poses = Tools.mat_pose_to_array(joints_path)
                point_clouds = Tools.mat_pt_cloud_to_array(pt_cloud_path)

                dim1, dim2, dim3 = point_clouds.shape
                annotated_clouds = np.empty((dim1, dim2, 1))

                print(f"Annotating: {pt_cloud_path} using: {joints_path}")
                for j in range(len(poses)):
                    if j % (len(poses) // 10) == 0:
                        print(f"{j} poses ouf of {len(poses)}")
                    pose = poses[j]
                    if j == len(point_clouds):
                        break
                    point_cloud = point_clouds[j]
                    annotated_clouds[j] = Annotation.annotate_ptcloud(point_cloud, pose)
                result_path = result_paths[i]
                Tools.array_to_np_file(annotated_clouds, result_path)
            except Exception as e:
                print(f"last index was {i}")
                error_logs.append((i, str(e)))

        return error_logs


if __name__ == '__main__':
    pose = Tools.np_file_to_array("../output/poses/pose0.npy")
    cloud = Tools.np_file_to_array("../output/pt_clouds/pt_cloud0.npy")
    annotation = Annotation.annotate_ptcloud(cloud, pose)
    Sketch.plot_annotated_ptcloud(cloud, annotation, ticks=True)
    Sketch.plot_skeleton(pose)
    



