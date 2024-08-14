import os.path

import numpy as np
from tools import Tools


class Normalisation:

    @staticmethod
    def center_ptclouds(pt_clouds):
        avg = pt_clouds.mean(axis=1, keepdims=True)
        return pt_clouds - avg

    @staticmethod
    def center(pt_clouds_paths, result_paths):
        for i in range(len(pt_clouds_paths)):
            pt_clouds = Normalisation.center_ptclouds(Tools.mat_pt_cloud_to_array(pt_clouds_paths[i]))
            np.save(result_paths[i], pt_clouds)

    @staticmethod
    def normalise_ptclouds(pt_cloud, min_, max_):  # normalise to -1, 1 scale
        return (2 * pt_cloud - (max_ + min_)) / (max_ - min_)

    @staticmethod
    def find_min_max(paths, mat=True):
        min_, max_ = None, None
        for path in paths:

            pt_clouds = Tools.mat_pt_cloud_to_array(path) if mat else Tools.np_file_to_array(path)

            min1 = pt_clouds.min((0, 1))
            if min_ is None:
                min_ = min1
            min_ = Normalisation.get_min_point(min_, min1)

            max1 = pt_clouds.max((0, 1))
            if max_ is None:
                max_ = max1
            max_ = Normalisation.get_max_point(max_, max1)

        return min_, max_

    @staticmethod
    def get_min_point(point0, point1):
        x1, y1, z1 = point1
        x0, y0, z0 = point0
        x = x1 if x1 < x0 else x0
        y = y1 if y1 < y0 else y0
        z = z1 if z1 < z0 else z0
        return np.array((x, y, z))

    @staticmethod
    def get_max_point(point0, point1):
        x1, y1, z1 = point1
        x0, y0, z0 = point0
        x = x1 if x1 > x0 else x0
        y = y1 if y1 > y0 else y0
        z = z1 if z1 > z0 else z0
        return np.array((x, y, z))

    @staticmethod
    def normalise(pt_cloud_paths, result_paths, mat=True):
        res = Normalisation.get_min_max_from_file()
        if res:
            min_, max_ = res
        else:
            min_, max_ = Normalisation.find_min_max(pt_cloud_paths, mat)
            print(f"min: {min_}, max: {max_}")
            Normalisation.save_min_max(min_, max_, "minmax.txt")

        for i in range(len(pt_cloud_paths)):
            pt_cloud_path = pt_cloud_paths[i]
            point_clouds = Tools.mat_pt_cloud_to_array(pt_cloud_path) if mat else Tools.np_file_to_array(pt_cloud_path)

            print("Normalising: " + pt_cloud_path)

            normalised_clouds = Normalisation.normalise_ptclouds(point_clouds, min_, max_)
            result_path = result_paths[i]
            Tools.array_to_np_file(normalised_clouds, result_path)

    @staticmethod
    def save_min_max(min_, max_, path):
        with open(path, "w") as f:
            print(f"min: {min_}, max: {max_}", file=f)

    @staticmethod
    def get_min_max_from_file():
        if os.path.exists("minmax.txt"):
            with open("minmax.txt") as f:
                line = f.readline()
                min_t, max_t = line.split(",")
                min_t = min_t[min_t.index("[") + 1:min_t.index("]")]
                max_t = max_t[min_t.index("[") + 1:min_t.index("]")]
                min_l = list(map(float, min_t.split()))
                max_l = list(map(float, max_t.split()))
                return np.array(min_l), np.array(max_l)
