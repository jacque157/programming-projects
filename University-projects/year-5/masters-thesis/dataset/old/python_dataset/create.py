import numpy as np
import os
import re
import scipy
import ntpath

MAT_ROOT = os.path.join('..', 'mat') # Path to .mat files of point clouds
DATASETS = ['EKUT', 'Eyes_Japan', 'ACCAD', 'CMU']

SKELETONS_ROOT = os.path.join('..') # Path to .obj files of skeletons
MASK = np.zeros(52)
MASK[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21]] = 1  # skeleton key points used  
MASK = MASK == 1

OUT_ROOT = os.path.join('..', 'python_dataset')


def read_key_points(path):
    vertices = []
    with open(path) as file:
        for line in file:
            if line[0] == 'v':
                _, x, y, z = line.split()
                x, y, z = float(x), float(y), float(z)
                vertices.append(np.array([x, y, z]))
    return np.array(vertices)

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def create_point_cloud(file_path, dataset_path):
    mat = scipy.io.loadmat(file_path)
    structured_point_cloud = mat['img']

    point_cloud = structured_point_cloud.reshape(-1, 3)
    store_zero_centered_min_max(dataset_path, point_cloud)
    
    file_name = path_leaf(file_path)
    sequence, body, camera = map(int, re.findall(r'\d+', file_name))
    if 'female' in file_path:
        out_folder = os.path.join(dataset_path,
                                  'female',
                                  f'sequence_{sequence}', 
                                  f'camera_{camera}')
    elif 'male' in file_path:
        out_folder = os.path.join(dataset_path,
                                  'male',
                                  f'sequence_{sequence}', 
                                  f'camera_{camera}')
    else:
        out_folder = os.path.join(dataset_path, 
                                f'sequence_{sequence}', 
                                f'camera_{camera}')
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_name = f'pose_{body}'
    out_path = os.path.join(out_folder, out_name)
    np.save(out_path, structured_point_cloud)

def store_zero_centered_min_max(dataset_path, point_cloud):
    point_cloud -= np.mean(point_cloud, axis=0)
    min_ = np.min(point_cloud, axis=0)
    max_ = np.max(point_cloud, axis=0)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    file = os.path.join(dataset_path, 'min_max.npy')
    if os.path.exists(file):
        min_max = np.load(file)
        current_min = min_max[0,:]
        current_max = min_max[1,:]
        min_ = np.min(np.stack([min_, current_min]), axis=0)
        max_ = np.max(np.stack([max_, current_max]), axis=0)
        
    min_max = np.stack([min_, max_])
    np.save(file, min_max)

def create_point_clouds(poses_folder, root):
    for file_name in os.listdir(poses_folder):
        file_path = os.path.join(poses_folder, file_name)
        if os.path.isfile(file_path):
            create_point_cloud(file_path, root)
        else:
            create_point_clouds(file_path, root)

def create_key_points(poses_folder, root, mask):
    for file_name in os.listdir(poses_folder):
        file_path = os.path.join(poses_folder, file_name)
        if os.path.isfile(file_path):
            if 'pose' in file_path:
                create_skeleton(file_path, root, mask)
        else:
            create_key_points(file_path, root, mask)
        
def create_skeleton(file_path, dataset_path, mask):
    key_points_all = read_key_points(file_path)
    key_points = key_points_all[mask]

    file_name = path_leaf(file_path)
    sequence, body = map(int, re.findall(r'\d+', file_name))
    if 'female' in file_path:
        out_folder = os.path.join(dataset_path,
                                  'female',
                                  f'sequence_{sequence}',
                                  'skeletons')
    elif 'male' in file_path:
        out_folder = os.path.join(dataset_path,
                                  'male',
                                  f'sequence_{sequence}',
                                  'skeletons')
    else:
        out_folder = os.path.join(dataset_path, 
                                f'sequence_{sequence}',
                                  'skeletons')
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    out_name = f'pose_{body}_skeleton'
    out_path = os.path.join(out_folder, out_name)
    np.save(out_path, key_points)


def enum_sequences(poses_folder):
    for sequence_folder in os.listdir(poses_folder):
        folder_path = os.path.join(poses_folder, sequence_folder)
        if os.path.isdir(folder_path):
            enum_sequence(folder_path)

def enum_sequence(folder_path):
    poses_dir = os.path.join(folder_path, 'camera_1') 
    number_of_poses = np.uint16([len(os.listdir(poses_dir))])
    out_path = os.path.join(folder_path, 'sequence_length')
    np.save(out_path, number_of_poses)

if __name__ == '__main__':
    for dataset in DATASETS:
        dataset_path = os.path.join(MAT_ROOT, dataset)

        male_dataset_path = os.path.join(dataset_path, 'male')    
        if os.path.exists(male_dataset_path):
            create_point_clouds(male_dataset_path, os.path.join(OUT_ROOT, dataset))
            enum_sequences(os.path.join(OUT_ROOT, dataset, 'male'))

        female_dataset_path = os.path.join(dataset_path, 'female')
        if os.path.exists(female_dataset_path):
            create_point_clouds(female_dataset_path, os.path.join(OUT_ROOT, dataset))
            enum_sequences(os.path.join(OUT_ROOT, dataset, 'female'))
        
        skeletons_path = os.path.join(SKELETONS_ROOT, dataset)

        male_dataset_path = os.path.join(skeletons_path, 'male')    
        if os.path.exists(male_dataset_path):
            create_key_points(male_dataset_path, os.path.join(OUT_ROOT, dataset), MASK)

        female_dataset_path = os.path.join(skeletons_path, 'female')
        if os.path.exists(female_dataset_path):
            create_key_points(female_dataset_path, os.path.join(OUT_ROOT, dataset), MASK)


