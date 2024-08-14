import numpy as np
import os 
import matplotlib.pyplot as plt
import torch


def find_global_min_max(root, names):
    best_min = np.array([float('inf'), float('inf'), float('inf')])
    best_max = np.array([float('-inf'), float('-inf'), float('-inf')])
    for name in names:
        path = os.path.join(root, name, 'min_max.npy')
        min_, max_ = np.load(path)
        mask = min_ < best_min
        best_min[mask] = min_[mask]

        mask = max_ > best_max
        best_max[mask] = max_[mask]
    return best_min, best_max

def find_projection_matrix(img):
    x = img[:,:,0]
    y = img[:,:,1]
    z = img[:,:,2]

    mask = np.logical_or(np.logical_or(x != 0, y != 0), z != 0)
    points = img[mask]    
    h, w, x = img.shape
    vu = np.mgrid[0:h, 0:w]

    px = vu[1][mask]
    py = vu[0][mask]

    sys = []
    for i in range(0, len(px), 10):
        x, y, z = points[i]
        u, v = px[i], py[i]
        sys.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        sys.append([0, 0, 0, 0, x, y, z, 1,  -v * x, -v * y, -v * z, -v])

    U, s, V = np.linalg.svd(sys)

    M = V[-1, :].reshape(3, 4)
    return M

def plot_body(img, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

    pt_cloud = img.reshape(-1, 3)
    pt_cloud = np.random.permutation(pt_cloud)[::10, :]
    ax.scatter(pt_cloud[:, 0], 
               pt_cloud[:, 1], 
               pt_cloud[:, 2], 
               marker='o', 
               color='blue', 
               alpha=0.1)
    return ax
    
def plot_skeleton(pose, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

    """pairs = [(11, 8), (8, 5), (5, 2), (2, 0), # their right leg
             (10, 7), (7, 4), (4, 1), (1, 0), # their left leg
             (0, 3), (3, 6), (6, 9), (9, 12), (12, 13), # spine
             (19, 17), (17, 15), (15, 12), # their right arm
             (18, 16), (16, 14), (14, 12)] # their left arm"""
    pairs = [(0, 2), (2, 5), (5, 8), (8, 11), # right leg
             (0, 1), (1, 4), (4, 7), (7, 10), # left leg
             (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), # spine
             (9, 14), (14, 17), (17, 19), (19, 21), # right arm
             (9, 13), (13, 16), (16, 18), (18, 20)] # left arm
    
    for (start, end) in pairs:
        x_start, y_start, z_start = pose[start, :]
        x_end, y_end, z_end = pose[end, :]
        ax.plot((x_start, x_end),
                (y_start, y_end),
                (z_start, z_end),
                marker='x', 
                color='red')
        
    for i, v in enumerate(pose):
        label = f'{i}' 
        ax.text(v[0], v[1], v[2], label)

    return ax

def load_poses(root, dataset, seq, camera):
    poses_path = os.path.join(root,
                             dataset,
                             'male',
                             f'sequence_{seq}',
                             f'camera_{camera}',
                             'poses.npz')
    if os.path.exists(poses_path):
        return np.load(poses_path)
    
    poses_path = os.path.join(root,
                             dataset,
                             'female',
                             f'sequence_{seq}',
                             f'camera_{camera}',
                             'poses.npz')
    if os.path.exists(poses_path):
        return np.load(poses_path)

def load_pose(root, dataset, seq, frame, camera):
    poses = load_poses(root, dataset, seq, camera)
    if poses:
        return poses[f'pose_{frame}']

def load_body(root, dataset, seq, frame, camera):
    return load_pose(root, dataset, seq, frame, camera)

def load_skeletons(root, dataset, seq):
    skeletons_path = os.path.join(root,
                                 dataset,
                                 'male',
                                 f'sequence_{seq}',
                                 'skeletons',
                                 'skeletons.npz')
    if os.path.exists(skeletons_path):
        skeletons = np.load(skeletons_path)
        return skeletons
    
    skeletons_path = os.path.join(root,
                                 dataset,
                                 'female',
                                 f'sequence_{seq}',
                                 'skeletons',
                                 'skeletons.npz')
    if os.path.exists(skeletons_path):
        skeletons = np.load(skeletons_path)
        return skeletons

def load_skeleton(root, dataset, seq, frame):
    skeletons = load_skeletons(root, dataset, seq)
    if skeletons:
        return skeletons[f'pose_{frame}_skeleton'] 
    
def load_number_of_frames(root, dataset, seq):
    length_path = os.path.join(root,
                               dataset,
                               'female',
                               f'sequence_{seq}',
                               'sequence_length.npy')
    if os.path.exists(length_path):
        return np.load(length_path)
    
    length_path = os.path.join(root,
                               dataset,
                               'male',
                               f'sequence_{seq}',
                               'sequence_length.npy')
    if os.path.exists(length_path):
        return np.load(length_path)

def load_segmentations(root, dataset, seq, camera):
    segmentations_path = os.path.join(root,
                                    dataset,
                                    'male',
                                    f'sequence_{seq}',
                                    f'camera_{camera}',
                                    f'segmentations.npz')
    if os.path.exists(segmentations_path):
        return np.load(segmentations_path)

    segmentations_path = os.path.join(root,
                                    dataset,
                                    'female',
                                    f'sequence_{seq}',
                                    f'camera_{camera}',
                                    f'segmentations.npz')
    if os.path.exists(segmentations_path):
        return np.load(segmentations_path)
    
def load_segmentation(root, dataset, seq, frame, camera):
    annotations = load_segmentations(root, dataset, seq, camera)
    if annotations:
        return annotations[f'pose_{frame}_segmentation']
    
def reconstruct_skeleton(relative_joints, centre=None):    
    skeleton = np.zeros(relative_joints.shape)
    centre = relative_joints[0]
    skeleton[0] = np.zeros(3) if centre is None else centre
    
    skeleton[2] = skeleton[0] + relative_joints[2]
    skeleton[5] = skeleton[2] + relative_joints[5]
    skeleton[8] = skeleton[5] + relative_joints[8]
    skeleton[11] = skeleton[8] + relative_joints[11]

    skeleton[1] = skeleton[0] + relative_joints[1]
    skeleton[4] = skeleton[1] + relative_joints[4]
    skeleton[7] = skeleton[4] + relative_joints[7]
    skeleton[10] = skeleton[7] + relative_joints[10]

    skeleton[3] = skeleton[0] + relative_joints[3]
    skeleton[6] = skeleton[3] + relative_joints[6]
    skeleton[9] = skeleton[6] + relative_joints[9]
    skeleton[12] = skeleton[9] + relative_joints[12]
    skeleton[15] = skeleton[12] + relative_joints[15]

    skeleton[14] = skeleton[9] + relative_joints[14]
    skeleton[17] = skeleton[14] + relative_joints[17]
    skeleton[19] = skeleton[17] + relative_joints[19]
    skeleton[21] = skeleton[19] + relative_joints[21]

    skeleton[13] = skeleton[9] + relative_joints[13]
    skeleton[16] = skeleton[13] + relative_joints[16]
    skeleton[18] = skeleton[16] + relative_joints[18]
    skeleton[20] = skeleton[18] + relative_joints[20]
    
    """skeleton[1] = skeleton[0] + relative_joints[1]
    skeleton[2] = skeleton[0] + relative_joints[2]
    skeleton[3] = skeleton[0] + relative_joints[3]
    skeleton[6] = skeleton[3] + relative_joints[6]
    skeleton[9] = skeleton[6] + relative_joints[9]
    skeleton[12] = skeleton[9] + relative_joints[12]
    skeleton[13] = skeleton[12] + relative_joints[13]

    skeleton[15] = skeleton[12] + relative_joints[15]
    skeleton[17] = skeleton[15] + relative_joints[17]
    skeleton[19] = skeleton[17] + relative_joints[19]

    skeleton[14] = skeleton[12] + relative_joints[14]
    skeleton[16] = skeleton[14] + relative_joints[16]
    skeleton[18] = skeleton[16] + relative_joints[18]

    skeleton[5] = skeleton[2] + relative_joints[5]
    skeleton[8] = skeleton[5] + relative_joints[8]
    skeleton[11] = skeleton[8] + relative_joints[11]

    skeleton[4] = skeleton[1] + relative_joints[4]
    skeleton[7] = skeleton[4] + relative_joints[7]
    skeleton[10] = skeleton[7] + relative_joints[10]"""
    return skeleton

def reconstruct_skeletons(relative_joints, centres=None):
    if type(relative_joints) is np.ndarray:
        skeletons = np.zeros(relative_joints.shape)
        s, j, c = skeletons.shape
        skeletons[:, 0] = np.zeros((s, c)) if centres is None else centres
    elif type(relative_joints) is torch.Tensor:
        skeletons = torch.zeros(relative_joints.shape, device=relative_joints.device)
        s, j, c = skeletons.shape
        skeletons[:, 0] = torch.zeros((s, c), device=relative_joints.device) if centres is None else centres
    
    skeletons[:, 2] = skeletons[:, 0] + relative_joints[:, 2]
    skeletons[:, 5] = skeletons[:, 2] + relative_joints[:, 5]
    skeletons[:, 8] = skeletons[:, 5] + relative_joints[:, 8]
    skeletons[:, 11] = skeletons[:, 8] + relative_joints[:, 11]

    skeletons[:, 1] = skeletons[:, 0] + relative_joints[:, 1]
    skeletons[:, 4] = skeletons[:, 1] + relative_joints[:, 4]
    skeletons[:, 7] = skeletons[:, 4] + relative_joints[:, 7]
    skeletons[:, 10] = skeletons[:, 7] + relative_joints[:, 10]

    skeletons[:, 3] = skeletons[:, 0] + relative_joints[:, 3]
    skeletons[:, 6] = skeletons[:, 3] + relative_joints[:, 6]
    skeletons[:, 9] = skeletons[:, 6] + relative_joints[:, 9]
    skeletons[:, 12] = skeletons[:, 9] + relative_joints[:, 12]
    skeletons[:, 15] = skeletons[:, 12] + relative_joints[:, 15]

    skeletons[:, 14] = skeletons[:, 9] + relative_joints[:, 14]
    skeletons[:, 17] = skeletons[:, 14] + relative_joints[:, 17]
    skeletons[:, 19] = skeletons[:, 17] + relative_joints[:, 19]
    skeletons[:, 21] = skeletons[:, 19] + relative_joints[:, 21]

    skeletons[:, 13] = skeletons[:, 9] + relative_joints[:, 13]
    skeletons[:, 16] = skeletons[:, 13] + relative_joints[:, 16]
    skeletons[:, 18] = skeletons[:, 16] + relative_joints[:, 18]
    skeletons[:, 20] = skeletons[:, 18] + relative_joints[:, 20]
    
    return skeletons

def annotate(pose, skeleton):
    difference = pose[:, :, None, :] - skeleton[None, None, :, :]
    distances = np.linalg.norm(difference, axis=3)
    annotation = np.argmin(distances, axis=2)

    ground_mask = np.abs(pose[:,:,2]) <= 10
    annotation[ground_mask] = np.max(annotation) + 1 # ground

    x = pose[:, :, 0]
    y = pose[:, :, 1]
    z = pose[:, :, 2]
    mask = np.logical_and(np.logical_and(x == 0, y == 0), z == 0)
    annotation[mask] = np.max(annotation) + 1 # background
    
    return np.uint8(annotation)

def plot_annotation(pose, annotation, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

    pt_cloud = pose.reshape(-1, 3)
    pt_cloud_annotation = annotation.reshape(-1, 1)

    pt_cloud = np.concatenate([pt_cloud, pt_cloud_annotation], axis=1)    
    pt_cloud = np.random.permutation(pt_cloud)[::10, :]
    
    for n in range(int(np.max(pt_cloud[:, 3])) + 1):
        mask = pt_cloud[:, 3] == n
        pt_cloud_n = pt_cloud[mask]
        ax.scatter(pt_cloud_n[:, 0], 
                pt_cloud_n[:, 1], 
                pt_cloud_n[:, 2], 
                marker='o', 
                alpha=1)
    return ax

def read_key_points(path):
    vertices = []
    with open(path) as file:
        for line in file:
            if line[0] == 'v':
                _, x, y, z = line.split()
                x, y, z = float(x), float(y), float(z)
                vertices.append(np.array([x, y, z]))
    return np.array(vertices, dtype=np.float32)

if __name__ == '__main__':
    """min_, max_ = find_global_min_max('Dataset', ['CMU'])
    print(min_)
    print(max_)
    print()

    min_, max_ = find_global_min_max('Dataset', ['ACCAD'])
    print(min_)
    print(max_)
    print()

    min_, max_ = find_global_min_max('Dataset', ['EKUT'])
    print(min_)
    print(max_)
    print()

    min_, max_ = find_global_min_max('Dataset', ['Eyes_Japan'])
    print(min_)
    print(max_)
    print()

    min_, max_ = find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
    print(min_)
    print(max_)
    print()"""

    body = load_body('Dataset', 'CMU', 20, 30, 2)
    ax = plot_body(body)
    skeleton = load_skeleton('Dataset', 'CMU', 20, 30)
    pth = os.path.join('D:', 'škola', 'matfyz', 'mgr', 'diplomovka', 'dataset',
                       'CMU', 'male', 'Part 1', 'seq20-body30_pose.obj')
    skeleton = read_key_points(pth)
    print(len(skeleton))
    plot_skeleton(skeleton, ax)
    #print(load_number_of_frames('Dataset', 'CMU', 20))
    an = annotate(body, skeleton)
    print(an.shape)
    plot_annotation(body, an)
    plt.show()


