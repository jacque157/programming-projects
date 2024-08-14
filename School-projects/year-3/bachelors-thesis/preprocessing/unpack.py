import os
from annotation import Annotation
from tools import Tools


def unpack_data(input_path, output_path):
    i = 0
    for file in os.scandir(input_path):
        file_name = file.name
        if file.is_dir():
            poses_path = os.path.join(input_path, file_name, "kinoptic_poses")
            clouds_path = os.path.join(input_path, file_name, "kinoptic_ptclouds")
            for j in range(len(list(os.scandir(poses_path)))):
                batched_poses_path = os.path.join(poses_path, f"pose_000000{j+1:02}.mat")
                batched_clouds_path = os.path.join(clouds_path, f"ptcloud_000000{j+1:02}.mat")
                try:
                    clouds = Tools.mat_pt_cloud_to_array(batched_clouds_path)
                    poses = Tools.mat_pose_to_array(batched_poses_path)

                    for k in range(len(poses)):
                        pose = poses[k]
                        pose_path = os.path.join(output_path, "poses", f"pose{i}.npy")
                        cloud = clouds[k]
                        cloud_path = os.path.join(output_path, "pt_clouds", f"pt_cloud{i}.npy")

                        try:
                            Tools.array_to_np_file(pose, pose_path)
                            Tools.array_to_np_file(cloud, cloud_path)
                            i += 1
                        except Exception as e:
                            print(e)
                    print('*' * 50)
                    print(batched_poses_path)
                    print(batched_clouds_path)
                    print("Done")
                    print('*' * 50)
                except Exception as e:
                    print(e)


def annotate_point_clouds(input_path, output_path):
    poses_path = os.path.join(input_path, "poses")
    clouds_path = os.path.join(input_path, "pt_clouds")
    annotations_path = os.path.join(output_path, "annotations")

    number_of_poses = len(list(os.scandir(poses_path)))
    for i in range(number_of_poses):
        pose_path = os.path.join(poses_path, f"pose{i}.npy")
        cloud_path = os.path.join(clouds_path, f"pt_cloud{i}.npy")
        annotation_path = os.path.join(annotations_path, f"gr_truth{i}.npy")

        cloud = Tools.np_file_to_array(cloud_path)
        pose = Tools.np_file_to_array(pose_path)
        annotation = Annotation.annotate_ptcloud(cloud, pose)

        Tools.array_to_np_file(annotation, annotation_path)

        if i % 1000 == 0:
            print(f"Annotated {i} poses out of {number_of_poses}.")


if __name__ == '__main__':
    # unpack_data("input", "output")
    annotate_point_clouds("../output", "../output")
