import os
import struct
import numpy as np
import json
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
from PIL import Image

INPUT_PATH = r"D:\datasetGenerator\ScanShoot-v1.2.0-win64\output\CMU"
OUTPUT_PATH = r"C:\Users\Erik Jakubovský\Documents\Diplomovka\codes_4\dataset\CMU"
NAME = "CMU"

HEATMAP_WIDTH = 128 #129
HEATMAP_HEIGHT = 128 #97
HEATMAP_FRACTION = 4
SIGMA = 2
KERNEL = 11

MAX_JOINTS = 22


POINTS = 2048
FX, FY, CX, CY = 400, 400, 258, 193 #256, 256 # one principal point was set incorrectly the generated heat maps are wrong the trained networks had mpjpe of 40mm, maybe with correct heat maps this can be imrpoved
# 400, 400, 258, 193 used for heatmaps 
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

def read_imgs(path):
    scans = len(os.listdir(path))
    imgs = []
    for scan in range(scans):
        scan_path = os.path.join(path, f'scan_{scan}.png')
        img = Image.open(scan_path)
        imgs.append(np.array(img))
    return np.array(imgs)

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

def load_imgs(input_path, sequence_number):
    sequence = []
    frame_number = 0
    while True:
        scan_folder = f"seq{sequence_number}-body{frame_number}"
        scan_folder_path = os.path.join(input_path, scan_folder)
        if not os.path.exists(scan_folder_path):
            break
        
        scans = read_imgs(scan_folder_path)
        if len(sequence) == 0:
            sequence = [[] for _ in range(len(scans))]
        for i, scan in enumerate(scans):
            sequence[i].append(scan)   
        frame_number += 1
    return np.array(sequence, dtype=np.uint8)

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

def store_imgs_as_npz(imgs, sequence_number, output_path):
    sequence_path = os.path.join(output_path, f"sequence_{sequence_number}")
    os.makedirs(sequence_path, exist_ok=True)
    for i, img in enumerate(imgs):
        camera_path = os.path.join(sequence_path, f"camera_{i + 1}")
        os.makedirs(camera_path, exist_ok=True)

        imgs_path = os.path.join(camera_path, f"imgs")
        name_img = {f"frame_{j}" : frame for j, frame in enumerate(img)}
        np.savez_compressed(imgs_path, **name_img)

def create_data(input_path, output_path):
    sequence_number = 1
    os.makedirs(output_path, exist_ok=True)
    tries = 0
    while tries < 100:
        print(f"processing sequence: {sequence_number}")
        scans = load_scans(input_path, sequence_number)
        skeletons = load_skeletons(input_path, sequence_number)
        imgs = load_imgs(input_path, sequence_number)

        if len(scans) == 0 or len(skeletons) == 0 or len(imgs) == 0:
            tries += 1
        else:
            store_scans_as_npz(scans, sequence_number - tries, output_path)
            store_skeletons_as_npz(skeletons, sequence_number - tries, output_path)
            store_imgs_as_npz(imgs, sequence_number - tries, output_path)

        sequence_number += 1

def load_structured_npz_point_clouds(path):
    data = np.load(path)
    return np.array([data[f"frame_{i}"] for i in range(len(data))], dtype=np.float32)

def load_img_npz(path):
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

def create_meta_data(input_path, camera_positions, camera_look_at_points, fx, fy, cx, cy, cameras=4, name='CMU'):
    meta_data = {}
    K = [[fx, 0, cx, 0],
         [0, fy, cy, 0],
         [0, 0, 1, 0]]
    meta_data["intrinsic_matrix"] = K
    Rts = []
    up = np.array((0, 0, 1))
    ts = []
    for position, target in zip(camera_positions, camera_look_at_points):
        ts.append([float(coord) for coord in position])
        Rt = look_at(position, target, up)
        Rt = [list(line) for line in Rt]
        Rts.append(Rt)
    meta_data["extrinsic_matrices"] = Rts
    meta_data["camera_positions"] = ts
    meta_data["name"] = name

    sequence_number = 1
    points_sum = np.zeros(3)
    points_max = np.ones(3, dtype=np.float64) * float('-inf')
    points_min = np.ones(3, dtype=np.float64) * float('inf')
    pixel_sum = np.zeros(3)
    pixel_min = np.zeros(3, dtype=np.float64)
    pixel_max = np.zeros(3, dtype=np.float64)
    cameras_sum = np.zeros(3)
    camera_points_min = np.zeros(3, dtype=np.float64)
    camera_points_max = np.zeros(3, dtype=np.float64)
    frames = 0

    while True:
        print(f"processing sequence: {sequence_number}")
        sequence_path = os.path.join(input_path, f'sequence_{sequence_number}')

        if not os.path.exists(sequence_path):
            break
        for camera in range(1, cameras + 1):
            camera_path = os.path.join(sequence_path, f'camera_{camera}')

            scans_path = os.path.join(camera_path, f'scans.npz')
            scans = load_structured_npz_point_clouds(scans_path)
            points_sum += np.sum(scans, (0, 1, 2))
            points_max = np.maximum(points_max, np.max(scans, (0, 1, 2)))
            points_min = np.minimum(points_min, np.min(scans, (0, 1, 2)))

            imgs_path = os.path.join(camera_path, f'imgs.npz')
            imgs = load_img_npz(imgs_path)
            pixel_sum += np.sum(imgs, (0, 1, 2))
            pixel_max = np.maximum(pixel_max, np.max(imgs, (0, 1, 2)))
            pixel_min = np.minimum(pixel_min, np.min(imgs, (0, 1, 2)))

            x = scans[:, :, :, 0]
            y = scans[:, :, :, 1]
            z = scans[:, :, :, 2]
            mask = np.logical_not(np.logical_and(np.logical_and(x == 0, y == 0), z == 0))
            flat_scans = scans[mask] #np.reshape(scans, (s * h * w, 3))
            flat_scans = np.concatenate((flat_scans, np.ones((len(flat_scans), 1))), 1)
            extrinsic_matrix = Rts[camera - 1]
            camera_space_points = (extrinsic_matrix @ flat_scans.T).T
            cameras_sum += np.sum(camera_space_points[:, :3], 0)
            camera_points_max = np.maximum(camera_points_max, np.max(camera_space_points[:, :3], 0))
            camera_points_min = np.minimum(camera_points_min, np.min(camera_space_points[:, :3], 0))
        s, h, w, c = scans.shape
        frames += s
        sequence_number += 1
    
    meta_data["avg_pixel"] = list(pixel_sum / (frames * cameras * h * w))
    meta_data["avg_point"] = list(points_sum / (frames * cameras * h * w))
    meta_data["avg_camera_point"] = list(cameras_sum / (frames * cameras * h * w))

    meta_data["pixel_sum"] = list(pixel_sum)
    meta_data["point_sum"] = list(points_sum)
    meta_data["camera_point_sum"] = list(cameras_sum)
    
    meta_data["min_pixel"] = list(pixel_min)
    meta_data["max_pixel"] = list(pixel_max)
    meta_data["min_point"] = list(points_min)
    meta_data["max_point"] = list(points_max)
    meta_data["min_camera_point"] = list(camera_points_min)
    meta_data["max_camera_point"] = list(camera_points_max)

    meta_data["min_pixel_corrected"] = list(pixel_min - meta_data["avg_pixel"])
    meta_data["max_pixel_corrected"] = list(pixel_max - meta_data["avg_pixel"])
    meta_data["min_point_corrected"] = list(points_min - meta_data["avg_point"])
    meta_data["max_point_corrected"] = list(points_max - meta_data["avg_point"])
    meta_data["min_camera_point_corrected"] = list(camera_points_min - meta_data["avg_camera_point"])
    meta_data["max_camera_point_corrected"] = list(camera_points_max - meta_data["avg_camera_point"])

    meta_data["sequences"] = sequence_number - 1
    meta_data["frames"] = frames
    
    
    os.makedirs(input_path, exist_ok=True)
    file_path = os.path.join(input_path, 'meta_data.json')
    with open(file_path, 'w') as file:
        json.dump(meta_data, file, indent=4)

def create_meta_data_std(input_path, cameras):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)

    points_avg = np.array(meta_data['avg_point'])[None, None, None, :]
    pixels_avg = np.array(meta_data['avg_pixel'])[None, None, None, :]
    cameras_avg = np.array(meta_data['avg_camera_point'])[None, :]

    points_dif_sum = np.zeros(3)
    pixels_dif_sum = np.zeros(3)
    cameras_dif_sum = np.zeros(3)

    Rts = meta_data["extrinsic_matrices"]
    
    frames = 0
    sequence_number = 1
    while True:
        print(f"processing sequence: {sequence_number}")
        sequence_path = os.path.join(input_path, f'sequence_{sequence_number}')
        if not os.path.exists(sequence_path):
            break
        for camera in range(1, cameras + 1):
            camera_path = os.path.join(sequence_path, f'camera_{camera}')

            scans_path = os.path.join(camera_path, f'scans.npz')
            scans = load_structured_npz_point_clouds(scans_path)
            points_dif_sum += np.sum((scans - points_avg) ** 2, (0, 1, 2))

            imgs_path = os.path.join(camera_path, f'imgs.npz')
            imgs = load_img_npz(imgs_path)
            pixels_dif_sum += np.sum((imgs - pixels_avg) ** 2, (0, 1, 2))

            x = scans[:, :, :, 0]
            y = scans[:, :, :, 1]
            z = scans[:, :, :, 2]
            mask = np.logical_not(np.logical_and(np.logical_and(x == 0, y == 0), z == 0))
            flat_scans = scans[mask] #np.reshape(scans, (s * h * w, 3))
            flat_scans = np.concatenate((flat_scans, np.ones((len(flat_scans), 1))), 1)
            extrinsic_matrix = Rts[camera - 1]
            camera_space_points = (extrinsic_matrix @ flat_scans.T).T

            a, b, c, d = scans.shape
            N = (a * b * c) - len(camera_space_points)
            cameras_dif_sum += np.sum((camera_space_points[:, :3] - cameras_avg) ** 2, 0)
            cameras_dif_sum += N * (cameras_avg ** 2).squeeze() # acount zeroes 
        s, h, w, c = scans.shape
        frames += s
        sequence_number += 1

    meta_data["var_pixel"] = list(pixels_dif_sum / (frames * cameras * h * w))
    meta_data["var_point"] = list(points_dif_sum / (frames * cameras * h * w))
    meta_data["var_camera_point"] = list(cameras_dif_sum / (frames * cameras * h * w))

    meta_data["std_pixel"] = list(np.sqrt(meta_data["var_pixel"]))
    meta_data["std_point"] = list(np.sqrt(meta_data["var_point"]))
    meta_data["std_camera_point"] = list(np.sqrt(meta_data["var_camera_point"]))
    
    os.makedirs(input_path, exist_ok=True)
    file_path = os.path.join(input_path, 'meta_data.json')
    with open(file_path, 'w') as file:
        json.dump(meta_data, file, indent=4)

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

def create_heatmaps(input_path, height, width, fraction, sigma, kernel_size, cameras):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)
    intrinsic_matrix = np.array(meta_data["intrinsic_matrix"])
                                
    for seq_number in tqdm(range(1, meta_data["sequences"] + 1)):
        sequence_path = os.path.join(input_path, f'sequence_{seq_number}')
        skeletons_path = os.path.join(sequence_path, 'skeletons.npz')
        skeletons = load_npz_skeletons(skeletons_path)
        s, n, c = skeletons.shape   
        
        for camera_index in range(1, cameras + 1):
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

def create_offsetmaps(input_path, height, width, fraction, cameras=4):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)
    intrinsic_matrix = np.array(meta_data["intrinsic_matrix"])
                                
    for seq_number in tqdm(range(1, meta_data["sequences"] + 1)):
        sequence_path = os.path.join(input_path, f'sequence_{seq_number}')
        skeletons_path = os.path.join(sequence_path, 'skeletons.npz')
        skeletons = load_npz_skeletons(skeletons_path)[:, 0 : 22, :]
        s, n, c = skeletons.shape   
        
        for camera_index in range(1, cameras + 1):
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

def create_depthtmaps(input_path, height, width, fraction, cameras=4):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)
    intrinsic_matrix = np.array(meta_data["intrinsic_matrix"])
                                
    for seq_number in tqdm(range(1, meta_data["sequences"] + 1)):
        sequence_path = os.path.join(input_path, f'sequence_{seq_number}')
        skeletons_path = os.path.join(sequence_path, 'skeletons.npz')
        skeletons = load_npz_skeletons(skeletons_path)[:, 0 : 22, :]
        s, n, c = skeletons.shape   
        
        for camera_index in range(1, cameras + 1):
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
    plt.show()"""
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

def generate_visibility_table(skeleton, scan, camera_intrinsics, camera_extrinsics):
    j, c = skeleton.shape
    h, w, c = scan.shape
    
    xs = scan[:, :, 0]
    ys = scan[:, :, 1]
    zs = scan[:, :, 2]
    mask = np.logical_and(np.logical_and(xs == 0, ys == 0), zs == 0)
    distances = np.sum((scan[:, :, None, :] - skeleton[None, None, :, :]) ** 2, 3)
    labels = np.argmin(distances, 2)
    labels[mask] = j

    skeleton_homo = np.concatenate((skeleton, np.ones((len(skeleton), 1))), 1)
    skeleton_2d = (camera_intrinsics @ camera_extrinsics @ skeleton_homo.T).T
    depths = skeleton_2d[:, -1]
    skeleton_2d = np.int32(np.round(skeleton_2d / depths[:, None]))[:, :2]
    visibility_map = np.zeros(j, dtype=np.uint8)

    for id_, (u,v) in enumerate(skeleton_2d):
        visibility_map[id_] = (labels[v, u] == id_)
        
    """img = np.zeros((h, w, 3))
    colours = np.random.rand(j + 1, 3)#[None, :] * np.linspace(0, 1, j + 1)[:, None]
    for row in range(h):
        for col in range(w):
            l = labels[row, col]
            img[row, col] = colours[l]
    plt.imshow(img)
    plt.show()
    
    img = np.zeros((h, w, 3))
    for (u, v) in skeleton_2d:
        img[v, u] = [255, 255, 255]

    plt.imshow(img)
    plt.show()
    assert 2 == 5"""
    return visibility_map

def create_visibility_tables(input_path, max_joints):
    meta_path = os.path.join(input_path, 'meta_data.json')
    with open(meta_path) as file:
        meta_data = json.load(file)

    camera_intrinsics = np.array(meta_data['intrinsic_matrix'], dtype=np.float32)
    for seq_number in tqdm(range(1, meta_data["sequences"] + 1)):
        sequence_path = os.path.join(input_path, f'sequence_{seq_number}')
        skeletons_path = os.path.join(sequence_path, 'skeletons.npz')
        skeletons = load_npz_skeletons(skeletons_path)[:, 0 : max_joints, :]
        s, n, c = skeletons.shape   
                    
        for camera_index in range(1, 5):
            visibility_table = []
            scans_path = os.path.join(sequence_path, f'camera_{camera_index}', 'scans.npz')
            scans = np.load(scans_path)
            scans = np.array([scans[f'frame_{i}'] for i in range(len(scans))])
            camera_extrinsics = np.array(meta_data['extrinsic_matrices'][camera_index - 1], dtype=np.float32)
            for skeleton, scan in zip(skeletons, scans):
                visibility_table.append(generate_visibility_table(skeleton, scan, camera_intrinsics, camera_extrinsics))

            visibility_table_path = os.path.join(sequence_path, f'camera_{camera_index}', 'visible_joints.npz')
            name_vis =  {f"frame_{i}" : vis for i, vis in enumerate(visibility_table)}
            np.savez_compressed(visibility_table_path, **name_vis)  

if __name__ == '__main__':
    #create_data(INPUT_PATH, OUTPUT_PATH)
    #create_meta_data(OUTPUT_PATH, CAMERA_POSITIONS, CAMERA_TARGETS, FX, FY, CX, CY, len(CAMERA_POSITIONS), NAME)
    #create_meta_data_std(OUTPUT_PATH, len(CAMERA_POSITIONS))
    #create_heatmaps(OUTPUT_PATH, HEATMAP_HEIGHT, HEATMAP_WIDTH, HEATMAP_FRACTION, SIGMA, KERNEL, len(CAMERA_POSITIONS))
    #create_offsetmaps(OUTPUT_PATH, HEATMAP_HEIGHT, HEATMAP_WIDTH, HEATMAP_FRACTION, len(CAMERA_POSITIONS))
    #create_depthtmaps(OUTPUT_PATH, HEATMAP_HEIGHT, HEATMAP_WIDTH, HEATMAP_FRACTION, len(CAMERA_POSITIONS))
    #create_segmentations(OUTPUT_PATH, MAX_JOINTS)
    create_visibility_tables(OUTPUT_PATH, MAX_JOINTS)
