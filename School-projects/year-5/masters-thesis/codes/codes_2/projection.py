from utils import *
from Dataset import *
from Transforms import *
import matplotlib.pyplot as plt
import cv2 as cv

DATASET_PATH = os.path.join('..', 'codes', 'Dataset')
MINMAX_PATH = os.path.join(DATASET_PATH, 'min_max.npy')
min_, max_ = np.load(MINMAX_PATH)#find_global_min_max('Dataset', ['CMU', 'ACCAD', 'EKUT', 'Eyes_Japan'])
transforms = transforms.Compose([ZeroCenter(), Rescale(min_, max_, -1, 1)])

def draw_lines(pose_2d, img, colour=255, thickness=3):
    pairs = [(0, 2), (2, 5), (5, 8), (8, 11), # right leg
             (0, 1), (1, 4), (4, 7), (7, 10), # left leg
             (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), # spine
             (9, 14), (14, 17), (17, 19), (19, 21), # right arm
             (9, 13), (13, 16), (16, 18), (18, 20)] # left arm
    img = np.uint8((img * 255 // 2) + (255 // 2))
    #img = np.uint8(img)
    for first, second in pairs:
        start = pose_2d[first]
        end = pose_2d[second]
        img = cv.line(img, start, end, colour, thickness)
    return img

dataset = Poses3D(DATASET_PATH, 'CMU', 'tr', None)
dataset_transformed = Poses3D(DATASET_PATH, 'CMU', 'tr', transforms)
data = dataset[0]
data_t = dataset_transformed[0]
imgs, poses = data['sequences'], data['key_points']
imgs_t = data_t['sequences']

for idx in range(0, len(imgs), 4):
    img, pose = imgs[idx], poses[idx]
    ax = plot_body(img)
    plot_skeleton(pose, ax)
    img_t = imgs_t[idx]
    pose = np.hstack((pose, np.ones((len(pose), 1))))
    proj = find_projection_matrix(img)
    
    
    skeleton_img = np.zeros((img.shape[0], img.shape[1]))

    pose_2d = ((proj @ pose.T).T)

    """x = img[:, :, 0]
    y = img[:, :, 1]
    z = img[:, :, 2]
    mask = np.logical_not(np.logical_and(x == 0, np.logical_and(y == 0, z == 0)))
    cld = img[mask]
    cld = np.concatenate((cld, np.ones((len(cld), 1))), 1)
    print(cld.shape)
    body_2d = ((proj @ cld.T).T) 
    w = body_2d[:, 2, None]
    body_2d = body_2d / w
    plot_body(img)

    proj_inv = np.linalg.pinv(proj)
    body_3d = np.random.permutation((proj_inv @ (body_2d * w).T).T)[::10, :]
    w = body_3d[:, 3, None]
    body_3d /= w
    print(body_3d)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(body_3d[:, 0], 
               body_3d[:, 1], 
               body_3d[:, 2], 
               marker='o', 
               color='blue', 
               alpha=0.1)
    
    plt.show()"""


    
    pose_2d = pose_2d[:, :2] / pose_2d[:, 2, None]
    pose_2d = np.int16(np.floor(pose_2d))


    """radius = 5

    for (x, y) in pose_2d:
        Y, X = np.ogrid[: img.shape[0], : img.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= radius**2
        skeleton_img[mask] = 255


    #skeleton_img[pose_2d] = 255
    """

    f, axarr = plt.subplots(4, 1)

    for i in range(3):
        img_i = img_t[:, :, i]
        #axarr[i].imshow(draw_lines(pose_2d, img_i), cmap='gray')
        axarr[i].imshow(img_i, cmap='gray')
        axarr[i].axis('off')

    skeleton_img = draw_lines(pose_2d, skeleton_img)
    axarr[3].imshow(skeleton_img, cmap='gray')
    plt.axis('off')
    plt.show()
