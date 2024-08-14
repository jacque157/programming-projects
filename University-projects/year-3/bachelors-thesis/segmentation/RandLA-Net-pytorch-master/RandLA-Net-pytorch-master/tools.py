import numpy as np
import scipy as sp
import scipy.io


class Tools:

    @staticmethod
    def extract_data(path, data_name):
        return sp.io.loadmat(path, struct_as_record=True, mat_dtype=False)[data_name][0]

    @staticmethod
    def mat_pt_cloud_to_array(path):
        pt_clouds = Tools.extract_data(path, "pclData")
        dim1 = pt_clouds.shape[0]
        dim2, dim3 = pt_clouds[0].shape
        array = np.empty((dim1, dim2, dim3))

        for i in range(dim1):
            array[i] = pt_clouds[i]

        return array

    @staticmethod
    def mat_pose_to_array(path):
        pose = Tools.extract_data(path, "poseData")
        dim1 = pose.shape[0]
        try:
            dim2, dim3 = pose[0][0][0][1].shape
        except IndexError:
            print("Odd pose")
            return np.empty(0)
        array = np.empty((dim1, dim2, dim3))

        for i in range(dim1):
            try:
                array[i] = pose[i][0][0][1]
            except IndexError:
                array.resize((i, dim2, dim3))
                print("Odd pose")
                return array

        return array

    @staticmethod
    def np_file_to_array(path):
        return np.load(path, allow_pickle=True)

    @staticmethod
    def array_to_np_file(array, path):
        np.save(path, array, True)
