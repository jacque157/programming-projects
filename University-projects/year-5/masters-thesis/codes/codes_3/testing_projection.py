import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

from utils import load_pose, load_poses, find_projection_matrix, plot_body

def find_parameters(structured_point_cloud, fx, fy):
    h, w, _ = structured_point_cloud.shape
    cx = w / 2
    cy = h / 2
    intrinsic_matrix = np.array(((fx, 0, cx),
                                 (0, fy, cy),
                                 (0, 0, 1)))
    projection_matrix = find_projection_matrix(structured_point_cloud)
    extrinsic_matrix = np.linalg.inv(intrinsic_matrix) @ projection_matrix

    return intrinsic_matrix, extrinsic_matrix

def unpad_point_cloud(structured_point_cloud, left_padding=8, right_padding=8, up_padding=52, down_padding=51):
    h, w, _ = structured_point_cloud.shape
    return structured_point_cloud[up_padding : h - down_padding, left_padding : w - right_padding]

def rads2degs(rads):
    return (rads * 360 / (2 * np.pi)) % 360

def rotation2angles(matrix, radians=True):
    if -0.9999 <= matrix[2, 0] <= 0.9999:
        theta1 = -np.arcsin(matrix[2, 0])
        theta2 = np.pi - theta1
        psi1 = np.arctan2(matrix[2, 1] / np.cos(theta1), matrix[2, 2] / np.cos(theta1))
        psi2 = np.arctan2(matrix[2, 1] / np.cos(theta2), matrix[2, 2] / np.cos(theta2))
        phi1 = np.arctan2(matrix[1, 0] / np.cos(theta1), matrix[0, 0] / np.cos(theta1))
        phi2 = np.arctan2(matrix[1, 0] / np.cos(theta2), matrix[0, 0] / np.cos(theta2))
        if radians:
            return (psi1, theta1, phi1), (psi2, theta2, phi2)
        else:
            return (rads2degs(psi1), rads2degs(theta1), rads2degs(phi1)), (rads2degs(psi2), rads2degs(theta2), rads2degs(phi2))
    else:
        phi = 0
        if matrix[2, 0] >= -0.9999:
            theta = np.pi / 2
            psi = phi + np.arctan2(matrix[0, 1], matrix[0, 2])
        else:
            theta = -np.pi / 2
            psi = -phi + np.arctan2(-matrix[0, 1], -matrix[0, 2])
        if radians:
            return psi, theta, phi
        else:
            return rads2degs(psi), rads2degs(theta), rads2degs(phi)

def structured2unstructured(structured_point_cloud):
    x = structured_point_cloud[:, :, 0]
    z = structured_point_cloud[:, :, 1]
    y = structured_point_cloud[:, :, 2]
    mask = np.logical_not(np.logical_and(x == 0, np.logical_and(y == 0, z == 0)))
    pt_cloud = structured_point_cloud[mask]
    return pt_cloud

def structured2depthmap(structured_point_cloud, intrinsic_matrix, extrinsic_matrix):
    pt_cloud = structured2unstructured(structured_point_cloud)
    pt_cloud = np.concatenate((pt_cloud, np.ones((len(pt_cloud), 1))), 1)
    projected = (intrinsic_matrix @ extrinsic_matrix @ pt_cloud.T).T
    ws = projected[:, -1, None]
    uv = projected / ws
    uv = np.uint16(np.round(uv))
    h, w, _ = structured_point_cloud.shape
    depth_map = np.zeros((h, w))
    depth_map[uv[:, 1], uv[:, 0]] = ws.flatten()
    return depth_map

def depthmap2unstructured(depthmap, intrinsic_matrix, extrinsic_matrix):
    return structured2unstructured(depthmap2structured(depthmap, intrinsic_matrix, extrinsic_matrix))

def depthmap2structured(depthmap, intrinsic_matrix, extrinsic_matrix):
    h, w = depthmap.shape
    vu = np.mgrid[0 : h, 0 : w]
    u = vu[1, :, :]
    v = vu[0, :, :]
    points_2d = np.stack((u, v, np.ones((h, w))), 2)
    mask = depthmap != 0
    ws = depthmap[mask]
    points_2d = points_2d[mask]
    rotation = extrinsic_matrix[:3, :3]
    translation = extrinsic_matrix[:3, 3]
    reprojection = (np.linalg.inv(rotation) @ ((np.linalg.inv(intrinsic_matrix) @ (np.float32(points_2d) * ws[:, None]).T) - translation[:, None])).T

    structured_pt_cloud = np.zeros((h, w, 3))
    for (x, y, z), (u, v, w) in zip(reprojection, points_2d):
        u, v = np.uint16(np.round(u / w)), np.uint16(np.round(v / w))
        structured_pt_cloud[v, u] = np.array([x, y, z])

    return structured_pt_cloud

def solve_pnp(structured_point_cloud, fx, fy):
    h, w, _ = structured_point_cloud.shape
    cx = w / 2
    cy = h / 2
    intrinsic_matrix = np.array(((fx, 0, cx),
                                 (0, fy, cy),
                                 (0, 0, 1)))
    x = structured_point_cloud[:, :, 0]
    z = structured_point_cloud[:, :, 1]
    y = structured_point_cloud[:, :, 2]
    mask = np.logical_not(np.logical_and(x == 0, np.logical_and(y == 0, z == 0)))
    points_3d = np.double(structured_point_cloud[mask])
    
    vu = np.mgrid[0 : h, 0 : w]

    u = vu[1, :, :][mask]
    v = vu[0, :, :][mask]
    point_2d = np.double(np.stack((u, v), -1))
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(points_3d,
                                                                point_2d,
                                                                intrinsic_matrix, 
                                                                dist_coeffs,
                                                                flags=0)
    if success:
        extrinsic_matrix = np.zeros((3, 4))
        return rotation_vector, translation_vector
    else:
        return None, None
    
def animate(root, dataset, sequence, camera):
    ims = []
    poses = load_poses(root, dataset, sequence, camera)
    fig, ax = plt.subplots()
    for i, pose in enumerate(poses):
        pose = poses[f'pose_{i}']
        x = pose[:, :, 0]
        y = pose[:, :, 1]
        z = pose[:, :, 2]
        mask = np.logical_not(np.logical_and(x == 0, np.logical_and(y == 0, z == 0)))
        im = ax.imshow(mask, cmap='gray', animated=True)
        if i == 0:
            ax.imshow(mask, cmap='gray')  # show an initial one first
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    writergif = animation.PillowWriter(fps=3)
    ani.save('filename.gif', writer=writergif)
    plt.show()

def animate_depthmap(root, dataset, sequence, camera):
    ims = []
    poses = load_poses(root, dataset, sequence, camera)
    fig, ax = plt.subplots()
    for i, pose in enumerate(poses):
        pose = poses[f'pose_{i}']
        pose = unpad_point_cloud(pose)
        h, w, _ = pose.shape
        projection_matrix = find_projection_matrix(pose)      
        x = pose[:, :, 0]
        y = pose[:, :, 1]
        z = pose[:, :, 2]
        mask = np.logical_not(np.logical_and(x == 0, np.logical_and(y == 0, z == 0)))
        points_3d = pose[mask]
        points_3d = np.concatenate((points_3d, np.ones((len(points_3d), 1))), 1)
        points_2d = (projection_matrix @ points_3d.T).T

        img = np.zeros((h, w))
        for (x, y, c) in points_2d:
            row = np.uint16(np.round(y / c))
            col = np.uint16(np.round(x / c))
            img[row, col] = np.abs(c)
        
        im = ax.imshow(img, cmap='gray', animated=True)
        if i == 0:
            ax.imshow(img, cmap='gray')  # show an initial one first
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    writergif = animation.PillowWriter(fps=3)
    ani.save('filename.gif', writer=writergif)
    plt.show()

if __name__ == '__main__':
    #animate_depthmap(root='../codes/Dataset', dataset='CMU', sequence=1, camera=1)
    pose = load_pose('../codes/Dataset', 'CMU', 1, 1, 1)
    original_image = unpad_point_cloud(pose)
    x = original_image[:, :, 0]
    y = original_image[:, :, 1]
    z = original_image[:, :, 2]

    print(x.shape)
    original_image = np.stack((x, y, z), 2)
    plot_body(original_image)
    plt.show()
    intrinsic_matrix, extrinsic_matrix = find_parameters(original_image, 3000, 3000)

    pt_cloud = structured2unstructured(original_image)

    depthmap = structured2depthmap(original_image, intrinsic_matrix, extrinsic_matrix)
    structured_point_cloud = depthmap2structured(depthmap, intrinsic_matrix, extrinsic_matrix)
    unstructured_point_cloud = structured2unstructured(structured_point_cloud)
    
    plot_body(unstructured_point_cloud)
    plt.show()


    x = original_image[:, :, 0]
    y = original_image[:, :, 1]
    z = original_image[:, :, 2]
    mask1 = np.logical_not(np.logical_and(x == 0, np.logical_and(y == 0, z == 0)))

    x = structured_point_cloud[:, :, 0]
    y = structured_point_cloud[:, :, 1]
    z = structured_point_cloud[:, :, 2]
    mask2 = np.logical_not(np.logical_and(x == 0, np.logical_and(y == 0, z == 0)))
    
    print(np.mean(np.sum((original_image[mask1] - structured_point_cloud[mask2]) ** 2, 1) ** 0.5))
    rotation = extrinsic_matrix[0 : 3, 0 : 3]
    #print(rotation2angles(rotation))

    def fun(seq, frame, camera):
        pose = load_pose('../codes/Dataset', 'CMU', seq, frame, camera)
        original_image = unpad_point_cloud(pose)
        x = original_image[:, :, 0]
        y = original_image[:, :, 1]
        z = original_image[:, :, 2]
        original_image = np.stack((x, y, z), 2)
        intrinsic_matrix, extrinsic_matrix = find_parameters(original_image, 1000, 1000)
        rotation = extrinsic_matrix[0 : 3, 0 : 3]
        translation = extrinsic_matrix[0 : 3, 3]
        print(rotation2angles(rotation, True))
        print(translation)
        rot, tranlation = solve_pnp(original_image, 3000, 3000)
        #print(rot)
        
    fun(1, 1, 1)
    fun(1, 1, 2)
    fun(1, 1, 3)
    fun(1, 1, 4)
    print()
    fun(1, 2, 1)
    fun(1, 2, 2)
    fun(1, 2, 3)
    fun(1, 2, 4)
