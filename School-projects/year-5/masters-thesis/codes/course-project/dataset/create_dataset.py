import os
import struct
import numpy as np
import json
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt

INPUT_PATH = r"D:\datasetGenerator\ScanShoot-v1.2.0-win64\output\CMU_dataset"
OUTPUT_PATH = r"C:\Users\Erik Jakubovský\Documents\HUpPV_projekt\dataset\CMU"
NAME = "CMU"

HEATMAP_WIDTH = 128 #129
HEATMAP_HEIGHT = 128 #97
HEATMAP_FRACTION = 4
SIGMA = 2
KERNEL = 11

MAX_JOINTS = 22

#INPUT_PATH = r"D:\datasetGenerator\ScanShoot-v1.2.0-win64\output\Eyes_Japan_dataset"
#OUTPUT_PATH = r"C:\Users\Erik Jakubovský\Documents\HUpPV_projekt\dataset\Eyes_Japan"
#NAME = "Eyes_Japan"

#INPUT_PATH = r"D:\datasetGenerator\ScanShoot-v1.2.0-win64\output\EKUT_dataset"
#OUTPUT_PATH = r"C:\Users\Erik Jakubovský\Documents\HUpPV_projekt\dataset\EKUT"
#NAME = "EKUT"

#INPUT_PATH = r"D:\datasetGenerator\ScanShoot-v1.2.0-win64\output\ACCAD_dataset"
#OUTPUT_PATH = r"C:\Users\Erik Jakubovský\Documents\HUpPV_projekt\dataset\ACCAD"
#NAME = "ACCAD"

POINTS = 2048
FX, FY, CX, CY = 400, 400, 258, 193
CAMERA_POSITIONS = np.array(((0, 2300, 2000),
                             (-2300, 0, 2000),
                             (0, -2300, 2000),
                             (2300, 0, 2000)))
CAMERA_TARGETS = np.array(((0, 0, 900),
                           (0, 0, 900),
                           (0, 0, 900),
                           (0, 0, 900)))

def read_dat(path):
    scans = []
    with open(path, mode='rb') as file:
        scan_count = int.from_bytes(file.read(4), byteorder='little')
        scan_width = int.from_bytes(file.read(4), byteorder='little')
        scan_height = int.from_bytes(file.read(4), byteorder='little')
        
        for i in range(scan_count):
            scan = np.zeros((scan_height, scan_width, 3), dtype=np.float32)
            for r in range(scan_height):
                for c in range(scan_width):
                    x = struct.unpack('f', file.read(4))[0]
                    y = struct.unpack('f', file.read(4))[0]
                    z = struct.unpack('f', file.read(4))[0]
                    scan[r, c] = [x, y, z]
            scans.append(scan)
    return np.array(scans, dtype=np.float32)
                
def read_pose(path):
    skeleton = []
    with open(path) as file:
        for line in file:
            if line[0] == 'v':
                x, y, z = map(float, line[2:].split())
                skeleton.append((x, y, z))
    return np.array(skeleton, dtype=np.float32)

def load_scans(input_path, sequence_number):
    sequence = []
    frame_number = 0
    while True:
        dat_file = f"seq{sequence_number}-body{frame_number}.dat"
        dat_file_path = os.path.join(input_path, dat_file)
        if not os.path.exists(dat_file_path):
            break
        
        scans = read_dat(dat_file_path)
        if len(sequence) == 0:
            sequence = [[] for _ in range(len(scans))]
        for i, scan in enumerate(scans):
            sequence[i].append(scan)   
        frame_number += 1
    return np.array(sequence, dtype=np.float32)
    
def load_skeletons(input_path, sequence_number):
    skeletons = []
    frame_number = 0
    while True:   
        skeleton_file = f"seq{sequence_number}-body{frame_number}_pose.obj"
        skeleton_file_path = os.path.join(input_path, skeleton_file)
        if not os.path.exists(skeleton_file_path):
            break

        skeleton = read_pose(skeleton_file_path)
        skeletons.append(skeleton) 
        frame_number += 1
    return np.array(skeletons, dtype=np.float32)

def create_point_clouds(sequences, n=2048):
    cameras, frames, height, width, _ =  sequences.shape
    point_clouds = []
    for i in range(frames):
        scans = sequences[:, i]
        point_clouds.append(create_pointcloud(scans, n))
    return point_clouds
    
def store_scans_as_npz(scans, sequence_number, output_path):
    sequence_path = os.path.join(output_path, f"sequence_{sequence_number}")
    os.makedirs(sequence_path, exist_ok=True)
    for i, sequence in enumerate(scans):
        camera_path = os.path.join(sequence_path, f"camera_{i + 1}")
        os.makedirs(camera_path, exist_ok=True)

        scans_path = os.path.join(camera_path, f"scans")
        name_scan = {f"frame_{j}" : frame for j, frame in enumerate(sequence)}
        np.savez_compressed(scans_path, **name_scan)

def store_skeletons_as_npz(skeletons, sequence_number, output_path):
    sequence_path = os.path.join(output_path, f"sequence_{sequence_number}")
    os.makedirs(sequence_path, exist_ok=True)

    skeletons_path = os.path.join(sequence_path, f"skeletons")
    name_skeleton = {f"skeleton_{j}" : skeleton for j, skeleton in enumerate(skeletons)}
    np.savez_compressed(skeletons_path, **name_skeleton)

def store_point_clouds_as_npz(point_clouds, sequence_number, output_path):
    sequence_path = os.path.join(output_path, f"sequence_{sequence_number}")
    os.makedirs(sequence_path, exist_ok=True)

    point_clouds_path = os.path.join(sequence_path, f"point_clouds")
    name_points = {f"point_cloud_{j}" : point_cloud for j, point_cloud in enumerate(point_clouds)}
    np.savez_compressed(point_clouds_path, **name_points)
           
def create_dataset(input_path, output_path, name, number_of_points, fx, fy, cx, cy, camera_positions, camera_look_at_points):
    sums = np.zeros(3, dtype=np.float64)
    frames = 0

    sequence_number = 1
    os.makedirs(output_path, exist_ok=True)

    while True:
        print(f"processing sequence: {sequence_number}")
        scans = load_scans(input_path, sequence_number)
        skeletons = load_skeletons(input_path, sequence_number)
        if len(scans) == 0 or len(skeletons) == 0:
            break
        point_clouds = create_point_clouds(scans, number_of_points)
        
        store_scans_as_npz(scans, sequence_number, output_path)
        store_skeletons_as_npz(skeletons, sequence_number, output_path)
        store_point_clouds_as_npz(point_clouds, sequence_number, output_path)
        
        sequence_number += 1
        sums[0] += np.sum(scans[:, :, :, :, 0])
        sums[1] += np.sum(scans[:, :, :, :, 1])
        sums[2] += np.sum(scans[:, :, :, :, 2])
        c, s, h, w, _ = scans.shape
        frames += s

    points = number_of_points * frames
    pixels = frames * h * w
    create_meta_data(output_path, name, sequence_number - 1, sums, frames, pixels, points, fx, fy, cx, cy, camera_positions, camera_look_at_points)
     
def create_meta_data(output_path, name, sequences, sums, frames, pixels, points, fx, fy, cx, cy, camera_positions, camera_look_at_points):
    K = [[fx, 0, cx, 0],
         [0, fy, cy, 0],
         [0, 0, 1, 0]]

    Rts = []
    up = np.array((0, 0, 1))
    for position, target in zip(camera_positions, camera_look_at_points):
        Rt = look_at(position, target, up)
        Rt = [list(line) for line in Rt]
        Rts.append(Rt)

    K = [list(line) for line in K]
    meta = {"name" : name,
            "sequences" : sequences,
            "frames" : int(frames),
            "average_pixel" : list(sums / pixels), # incorrect
            "average_point" : list(sums / points), # incorrect
            "min_pixel" : None, "max_pixel" : None,
            "min_point" : None, "max_point" : None,
            "intrinsic_matrix" : K, "extrinsic_matrices" : Rts}
    
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, 'meta_data.json')
    with open(file_path, 'w') as file:
        json.dump(meta, file, indent=4)

def update_mean(input_path):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)
    points_sum = np.zeros(3, dtype=np.float64)
    pixel_sum = np.zeros(3, dtype=np.float64)
    camera_points_sum = np.zeros(3, dtype=np.float64)
    number_of_points = 0
    number_of_pixels = 0
    
    for seq_number in tqdm(range(1, meta_data["sequences"] + 1)):
        sequence_path = os.path.join(input_path, f'sequence_{seq_number}')
        point_cloud_path = os.path.join(sequence_path, 'point_clouds.npz')
        point_cloud = np.load(point_cloud_path)
        point_cloud = np.array([point_cloud[f'point_cloud_{i}'] for i in range(len(point_cloud))])
        assert len(point_cloud.shape) == 3

        s, n, c = point_cloud.shape
        number_of_points += s * n
        points_sum[0] += np.sum(point_cloud[:, :, 0])
        points_sum[1] += np.sum(point_cloud[:, :, 1])
        points_sum[2] += np.sum(point_cloud[:, :, 2])

        for camera_index in range(1, 5):
            extrinsic_matrix = np.array(meta_data["extrinsic_matrices"][camera_index - 1])
            scans_path = os.path.join(sequence_path, f'camera_{camera_index}', 'scans.npz')
            scans = np.load(scans_path)
            scans = np.array([scans[f'frame_{i}'] for i in range(len(scans))])
            assert len(scans.shape) == 4

            s, h, w, c = scans.shape
            number_of_pixels += s * h * w
            pixel_sum[0] = np.sum(scans[:, :, :, 0])
            pixel_sum[1] = np.sum(scans[:, :, :, 1])
            pixel_sum[2] = np.sum(scans[:, :, :, 2])

            x = scans[:, :, :, 0]
            y = scans[:, :, :, 1]
            z = scans[:, :, :, 2]
            mask = np.logical_not(np.logical_and(np.logical_and(x == 0, y == 0), z == 0))
            flat_scans = scans[mask] #np.reshape(scans, (s * h * w, 3))
            flat_scans = np.concatenate((flat_scans, np.ones((len(flat_scans), 1))), 1)
            camera_space_points = (extrinsic_matrix @ flat_scans.T).T
            camera_points_sum[0] += np.sum(camera_space_points[:, 0])
            camera_points_sum[1] += np.sum(camera_space_points[:, 1])
            camera_points_sum[2] += np.sum(camera_space_points[:, 2])

    meta_data["average_point"] = list(points_sum / number_of_points)
    meta_data["average_pixel"] = list(pixel_sum / number_of_pixels)
    meta_data["average_camera_point"] = list(camera_points_sum / number_of_pixels)    
    file_path = os.path.join(input_path, 'meta_data.json')
    with open(file_path, 'w') as file:
        json.dump(meta_data, file, indent=4)

def update_extremes(input_path):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)
    points_mean = np.array(meta_data["average_point"])
    points_max = np.ones(3, dtype=np.float64) * float('-inf')
    points_min = np.ones(3, dtype=np.float64) * float('inf')
    pixel_mean = np.array(meta_data["average_pixel"])
    pixel_min = np.zeros(3, dtype=np.float64)
    pixel_max = np.zeros(3, dtype=np.float64)
    camera_points_mean = np.array(meta_data["average_camera_point"])
    camera_points_min = np.zeros(3, dtype=np.float64)
    camera_points_max = np.zeros(3, dtype=np.float64)
    
    for seq_number in tqdm(range(1, meta_data["sequences"] + 1)):
        sequence_path = os.path.join(input_path, f'sequence_{seq_number}')
        point_cloud_path = os.path.join(sequence_path, 'point_clouds.npz')
        point_cloud = np.load(point_cloud_path)
        point_cloud = np.array([point_cloud[f'point_cloud_{i}'] for i in range(len(point_cloud))])
        assert len(point_cloud.shape) == 3

        point_cloud_centered = point_cloud - points_mean
        points_max = np.maximum(points_max, np.max(point_cloud_centered, (0, 1)))
        points_min = np.minimum(points_min, np.min(point_cloud_centered, (0, 1)))

        for camera_index in range(1, 5):
            extrinsic_matrix = np.array(meta_data["extrinsic_matrices"][camera_index - 1])
            scans_path = os.path.join(sequence_path, f'camera_{camera_index}', 'scans.npz')
            scans = np.load(scans_path)
            scans = np.array([scans[f'frame_{i}'] for i in range(len(scans))])
            assert len(scans.shape) == 4

            pixels_centered = scans - pixel_mean
            pixel_max = np.maximum(pixel_max, np.max(pixels_centered, (0, 1, 2)))
            pixel_min = np.minimum(pixel_min, np.min(pixels_centered, (0, 1, 2)))

            x = scans[:, :, :, 0]
            y = scans[:, :, :, 1]
            z = scans[:, :, :, 2]
            mask = np.logical_not(np.logical_and(np.logical_and(x == 0, y == 0), z == 0))
            flat_scans = scans[mask] #np.reshape(scans, (s * h * w, 3))
            flat_scans = np.concatenate((flat_scans, np.ones((len(flat_scans), 1))), 1)
            camera_space_points = (extrinsic_matrix @ flat_scans.T).T
            camera_space_points_centered = camera_space_points[:, :3] - camera_points_mean
            assert len(camera_space_points.shape) == 2
            camera_points_min = np.minimum(camera_points_min, np.min(camera_space_points_centered, 0))
            camera_points_max = np.maximum(camera_points_max, np.max(camera_space_points_centered, 0))

    meta_data["min_pixel"] = list(pixel_min)
    meta_data["max_pixel"] = list(pixel_max)
    meta_data["min_point"] = list(points_min)
    meta_data["max_point"] = list(points_max)
    meta_data["min_camera_point"] = list(camera_points_min)
    meta_data["max_camera_point"] = list(camera_points_max)
    
    file_path = os.path.join(input_path, 'meta_data.json')
    with open(file_path, 'w') as file:
        json.dump(meta_data, file, indent=4)
        
def create_pointcloud(views, n=2048):
    xs = views[:, :, :, 0]
    ys = views[:, :, :, 1]
    zs = views[:, :, :, 2]
    mask = np.logical_not(np.logical_and(np.logical_and(xs == 0, ys == 0), zs == 0))
    points = views[mask]
    return farthest_point_sampling(points, n)
    
def farthest_point_sampling(points, n):
    # https://minibatchai.com/2021/08/07/FPS.html
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]

def load_structured_npz_point_clouds(path):
    data = np.load(path)
    return np.array([data[f"frame_{i}"] for i in range(len(data))], dtype=np.float32)

def load_npz_skeletons(path):
    data = np.load(path)
    return np.array([data[f"skeleton_{i}"] for i in range(len(data))], dtype=np.float32)

def normalise(vector):
    norm = np.sqrt(np.sum(vector ** 2, -1))
    return vector / norm

def look_at(eye, centre, up):
    # https://ksimek.github.io/2012/08/22/extrinsic/
    camera_forward = normalise(centre - eye)
    camera_right = normalise(np.cross(camera_forward, up))
    camera_up = np.cross(camera_right, camera_forward)
    rotation =  np.stack((-camera_right,
                          camera_up,
                          -camera_forward), 0)
    translation = -rotation @ eye
    view_matrix = np.eye(4)
    view_matrix[0:3] = np.concatenate((rotation, translation[:, None]), 1)
    return view_matrix
    
def structured_point_cloud_to_depth_map(structured_point_cloud, intrinsic_matrix, extrinsic_matrix):
    height, width, _ = structured_point_cloud.shape
    depth_map = np.zeros((height, width))

    xs = structured_point_cloud[:, :, 0]
    ys = structured_point_cloud[:, :, 0]
    zs = structured_point_cloud[:, :, 0]
    mask = np.logical_not(np.logical_and(np.logical_and(xs == 0, ys == 0), zs == 0))

    points = structured_point_cloud[mask]
    points = np.concatenate((points, np.ones((len(points), 1))), 1)
    uvw = (intrinsic_matrix @ extrinsic_matrix @ points.T).T

    for (u, v, w) in uvw:
        c = int(np.round(u / w))
        r = int(np.round(v / w))
        if 0 <= c < width and 0 <= r < height:
           depth_map[r, c] = w

    return np.abs(depth_map)

"""def generate_heatmap(x, y, height, width, sigma, kernel_size):
    assert 0 <= x < width
    assert 0 <= y < height
    kernel_1D = signal.windows.gaussian(kernel_size, std=sigma)[:, None]
    kernel_2D = np.outer(kernel_1D, kernel_1D)

    heat_map = np.zeros((height, width))
    
    heatmap_y_start = y - (kernel_size // 2)
    heatmap_y_end = heatmap_y_start + kernel_size
    kernel_y_start = 0
    if heatmap_y_start < 0:
        kernel_y_start = abs(heatmap_y_start)
        heatmap_y_start = 0

    kernel_y_end = kernel_size
    if heatmap_y_end > height:
        kernel_y_end = kernel_size - (heatmap_y_end - height) 
        heatmap_y_end = height

    heatmap_x_start = x - (kernel_size // 2)
    heatmap_x_end = heatmap_x_start + kernel_size
    kernel_x_start = 0
    if heatmap_x_start < 0:
        kernel_x_start = abs(heatmap_x_start)
        heatmap_x_start = 0

    kernel_x_end = kernel_size
    if heatmap_x_end > width:
        kernel_x_end = kernel_size - (heatmap_x_end - width) 
        heatmap_x_end = width
    
    heat_map[heatmap_y_start : heatmap_y_end,
             heatmap_x_start : heatmap_x_end] = kernel_2D[kernel_y_start : kernel_y_end,
                                                          kernel_x_start : kernel_x_end]
    return heat_map
"""

def generate_heatmap(x, y, height, width, sigma, kernel_size):
    kernel_1D = signal.windows.gaussian(kernel_size, std=sigma)[:, None]
    kernel_2D = np.outer(kernel_1D, kernel_1D)

    heat_map = np.zeros((height + (2 * kernel_size), width + (2 * kernel_size)))
    
    heatmap_y_start = kernel_size + y - (kernel_size // 2)
    heatmap_y_end = heatmap_y_start + kernel_size
    heatmap_x_start = kernel_size + x - (kernel_size // 2)
    heatmap_x_end = heatmap_x_start + kernel_size

    h, w = heat_map.shape
    if heatmap_y_start >= 0 and heatmap_y_end < h and heatmap_x_start >= 0 and heatmap_x_end < w:
        heat_map[heatmap_y_start : heatmap_y_end,
                 heatmap_x_start : heatmap_x_end] = kernel_2D
    heat_map = heat_map[kernel_size : h - kernel_size, kernel_size : w - kernel_size]
    max_ = np.max(heat_map)
    if max_ != 0:
        heat_map /= max_
    #heat_map -= 1e-4
    #heat_map = np.abs(heat_map)
    return heat_map

def create_heatmaps(input_path, height, width, fraction, sigma, kernel_size):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)
    intrinsic_matrix = np.array(meta_data["intrinsic_matrix"])
                                
    for seq_number in tqdm(range(1, meta_data["sequences"] + 1)):
        sequence_path = os.path.join(input_path, f'sequence_{seq_number}')
        skeletons_path = os.path.join(sequence_path, 'skeletons.npz')
        skeletons = load_npz_skeletons(skeletons_path)
        s, n, c = skeletons.shape   
        
        for camera_index in range(1, 5):
            extrinsic_matrix = np.array(meta_data["extrinsic_matrices"][camera_index - 1])
            skeletons_heat_maps = []
            
            for skeleton in skeletons:
                skeleton = np.concatenate((skeleton, np.ones((n, 1))), 1)
                skeleton_2d = (intrinsic_matrix @ extrinsic_matrix @ skeleton.T).T
                depths = skeleton_2d[:, -1]
                skeleton_2d =  skeleton_2d / depths[:, None]
                skeleton_2d /= fraction
                skeleton_2d = np.int32(skeleton_2d) #np.int32(np.round(skeleton_2d))
                heat_maps = []
                for x, y, _ in skeleton_2d:
                    heat_maps.append(generate_heatmap(x, y, height, width, sigma, kernel_size))
                skeletons_heat_maps.append(np.stack(heat_maps, 2))
            skeletons_heat_maps = np.array(skeletons_heat_maps)
            hs, hh, hw, hn = skeletons_heat_maps.shape
            assert hs == s
            assert hh == height
            assert hw == width
            assert hn == n
                 
            heat_maps_path = os.path.join(sequence_path, f'camera_{camera_index}', 'heat_maps.npz')
            name_heat_maps = {f"heat_maps_{i}" : hm for i, hm in enumerate(skeletons_heat_maps)}
            np.savez_compressed(heat_maps_path, **name_heat_maps)           

def create_offsetmaps(input_path, height, width, fraction):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)
    intrinsic_matrix = np.array(meta_data["intrinsic_matrix"])
                                
    for seq_number in tqdm(range(1, meta_data["sequences"] + 1)):
        sequence_path = os.path.join(input_path, f'sequence_{seq_number}')
        skeletons_path = os.path.join(sequence_path, 'skeletons.npz')
        skeletons = load_npz_skeletons(skeletons_path)[:, 0 : 22, :]
        s, n, c = skeletons.shape   
        
        for camera_index in range(1, 5):
            extrinsic_matrix = np.array(meta_data["extrinsic_matrices"][camera_index - 1])
            skeletons_offset_maps = []
            
            for skeleton in skeletons:
                skeleton = np.concatenate((skeleton, np.ones((n, 1))), 1)
                skeleton_2d = (intrinsic_matrix @ extrinsic_matrix @ skeleton.T).T
                depths = skeleton_2d[:, -1]
                skeleton_2d_gt =  skeleton_2d / depths[:, None]
                skeleton_2d = skeleton_2d_gt / fraction
                skeleton_2d = np.int32(skeleton_2d) #np.int32(np.round(skeleton_2d))
                map_ = np.zeros((height, width, 2))
                for (x, y, _), (x_gt, y_gt, _) in zip(skeleton_2d, skeleton_2d_gt):
                    if 0 <= x < width and 0 <= y < height:
                        offset_x = (x_gt / fraction) - float(x)
                        offset_y = (y_gt / fraction) - float(y)
                        #print('gt pos', x_gt, y_gt)
                        #print('pos', x, y)
                        #print('offset', offset_x, offset_y)
                        map_[y, x, 0] = offset_x
                        map_[y, x, 1] = offset_y
                skeletons_offset_maps.append(map_)
            skeletons_offset_maps = np.array(skeletons_offset_maps)
            hs, hh, hw, hn = skeletons_offset_maps.shape
            assert hs == s
            assert hh == height
            assert hw == width
            assert hn == 2
                 
            offset_maps_path = os.path.join(sequence_path, f'camera_{camera_index}', 'skeletons_offset_maps.npz')
            name_offset_maps = {f"offset_maps_{i}" : hm for i, hm in enumerate(skeletons_offset_maps)}
            np.savez_compressed(offset_maps_path, **name_offset_maps)   

def create_depthtmaps(input_path, height, width, fraction):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)
    intrinsic_matrix = np.array(meta_data["intrinsic_matrix"])
                                
    for seq_number in tqdm(range(1, meta_data["sequences"] + 1)):
        sequence_path = os.path.join(input_path, f'sequence_{seq_number}')
        skeletons_path = os.path.join(sequence_path, 'skeletons.npz')
        skeletons = load_npz_skeletons(skeletons_path)[:, 0 : 22, :]
        s, n, c = skeletons.shape   
        
        for camera_index in range(1, 5):
            extrinsic_matrix = np.array(meta_data["extrinsic_matrices"][camera_index - 1])
            skeletons_depth_maps = []
            
            for skeleton in skeletons:
                skeleton = np.concatenate((skeleton, np.ones((n, 1))), 1)
                skeleton_2d = (intrinsic_matrix @ extrinsic_matrix @ skeleton.T).T
                depths = skeleton_2d[:, -1]
                skeleton_2d =  skeleton_2d / depths[:, None]
                skeleton_2d = skeleton_2d / fraction
                skeleton_2d = np.int32(skeleton_2d) #np.int32(np.round(skeleton_2d))
                map_ = np.zeros((height, width, 1))
                for (x, y, _), d in zip(skeleton_2d, depths):
                    if 0 <= x < width and 0 <= y < height:
                        map_[y, x, 0] = d
                skeletons_depth_maps.append(map_)
            skeletons_depth_maps = np.array(skeletons_depth_maps)
            hs, hh, hw, hn = skeletons_depth_maps.shape
            assert hs == s
            assert hh == height
            assert hw == width
            assert hn == 1
                 
            depth_maps_path = os.path.join(sequence_path, f'camera_{camera_index}', 'skeletons_depth_maps.npz')
            name_depth_maps = {f"depth_maps_{i}" : hm for i, hm in enumerate(skeletons_depth_maps)}
            np.savez_compressed(depth_maps_path, **name_depth_maps)  

def fix_camera_matrix(input_path, camera_positions, camera_look_at_points, fx, fy, cx, cy):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)

    K = [[fx, 0, cx, 0],
         [0, fy, cy, 0],
         [0, 0, 1, 0]]

    Rts = []
    up = np.array((0, 0, 1))
    for position, target in zip(camera_positions, camera_look_at_points):
        Rt = look_at(position, target, up)
        Rt = [list(line) for line in Rt]
        Rts.append(Rt)

    K = [list(line) for line in K]
    meta_data["intrinsic_matrix"] = K
    meta_data["extrinsic_matrices"] = Rts
    os.makedirs(input_path, exist_ok=True)
    file_path = os.path.join(input_path, 'meta_data.json')
    with open(file_path, 'w') as file:
        json.dump(meta_data, file, indent=4)

def generate_segmentation(skeleton, scan):
    j, c = skeleton.shape
    h, w, c = scan.shape
    segmentation = np.zeros((h, w, j + 1), dtype=np.uint8)

    xs = scan[:, :, 0]
    ys = scan[:, :, 1]
    zs = scan[:, :, 2]
    mask = np.logical_and(np.logical_and(xs == 0, ys == 0), zs == 0)
    distances = np.sum((scan[:, :, None, :] - skeleton[None, None, :, :]) ** 2, 3)
    labels = np.argmin(distances, 2)
    labels[mask] = j

    for i in range(j + 1):
        mask = labels == i
        segmentation[mask, i] = 1

    """img = np.zeros((h, w, 3))
    colours = np.random.rand(j + 1, 3)#[None, :] * np.linspace(0, 1, j + 1)[:, None]
    for row in range(h):
        for col in range(w):
            l = np.argmax(segmentation[row, col])
            img[row, col] = colours[l]

    plt.imshow(img)
    plt.show()
            
    print(segmentation.shape)
    print(np.argmax(segmentation, -1))"""
    """for row in range(h):
        for col in range(w):
            point = scan[row, col]
            if np.all(point == [0, 0, 0]):
                segmentation[-1] = 1
            else:
                distances = np.sum((skeleton - point[None, :]) ** 2, 1)
                best = np.argmin(distances, 0)
                assert type(best) is np.int64
                segmentation[best] = 1"""
    return segmentation

def create_segmentations(input_path, max_joints):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)
        
    for seq_number in tqdm(range(1, meta_data["sequences"] + 1)):
        sequence_path = os.path.join(input_path, f'sequence_{seq_number}')
        skeletons_path = os.path.join(sequence_path, 'skeletons.npz')
        skeletons = load_npz_skeletons(skeletons_path)[:, 0 : max_joints, :]
        s, n, c = skeletons.shape   
                    
        for camera_index in range(1, 5):
            segmentations = []
            scans_path = os.path.join(sequence_path, f'camera_{camera_index}', 'scans.npz')
            scans = np.load(scans_path)
            scans = np.array([scans[f'frame_{i}'] for i in range(len(scans))])
            
            for skeleton, scan in zip(skeletons, scans):
                segmentations.append(generate_segmentation(skeleton, scan))

            segmentation_path = os.path.join(sequence_path, f'camera_{camera_index}', 'segmentations.npz')
            name_segmentation =  {f"segmentation_{i}" : seg for i, seg in enumerate(segmentations)}
            np.savez_compressed(segmentation_path, **name_segmentation)  
                
           
if __name__ == '__main__':
    #create_dataset(INPUT_PATH, OUTPUT_PATH, NAME, POINTS, FX, FY, CX, CY, CAMERA_POSITIONS, CAMERA_TARGETS)
    #fix_camera_matrix(OUTPUT_PATH, CAMERA_POSITIONS, CAMERA_TARGETS, FX, FY, CX, CY)
    #update_mean(OUTPUT_PATH)
    #update_extremes(OUTPUT_PATH)
    #create_heatmaps(OUTPUT_PATH, HEATMAP_HEIGHT, HEATMAP_WIDTH, HEATMAP_FRACTION, SIGMA, KERNEL)
    #create_offsetmaps(OUTPUT_PATH, HEATMAP_HEIGHT, HEATMAP_WIDTH, HEATMAP_FRACTION)
    #create_depthtmaps(OUTPUT_PATH, HEATMAP_HEIGHT, HEATMAP_WIDTH, HEATMAP_FRACTION)
    create_segmentations(OUTPUT_PATH, MAX_JOINTS)
