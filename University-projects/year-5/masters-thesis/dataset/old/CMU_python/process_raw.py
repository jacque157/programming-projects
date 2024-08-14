import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io  

def annotate(pose, skeleton):
    difference = pose[:, :, None, :] - skeleton[None, None, :, :]
    distances = np.linalg.norm(difference, axis=3)
    annotation = np.argmin(distances, axis=2)

    return annotation

def read_pose(path):
    vertices = []
    with open(path) as file:
        for line in file:
            if line[0] == 'v':
                _, x, y, z = line.split()
                x, y, z = float(x), float(y), float(z)
                vertices.append(np.array([x, y, z]))
    return np.array(vertices)

def plot_pose(pose, ax):
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], marker='x', color='red')

    for i, v in enumerate(pose):
        label = f'{i}' 
        ax.text(v[0], v[1], v[2], label)
    
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

file = '../mat/CMU/female/Part 1/seq22-body0_point_cloud_4.mat'
pose = read_pose('../CMU/female/Part 1/seq22-body0_pose.obj')
#file = '../mat/CMU/male/Part 1/seq1-body0_point_cloud_4.mat'
#pose = read_pose('../CMU/male/Part 1/seq1-body0_pose.obj')
mat = scipy.io.loadmat(file)
img = mat['img']
i = np.random.randint(0, 257 - 224)
j = np.random.randint(0, 344 - 224)
img = img[i:i+224, j:j+224, :]
pt_cloud = img.reshape(-1, 3)[::20, :]
#pt_cloud = mat['img'].reshape(-1, 3)[::20, :]
print(img.shape)
#sub_cloud = img[200:250, 200:250, :]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#ax.scatter(sub_cloud[:, 0], sub_cloud[:, 1], sub_cloud[:, 2], marker='^', color='blue')
ax.scatter(pt_cloud[:, 0], pt_cloud[:, 1], pt_cloud[:, 2], marker='o', color='blue', alpha=0.1)
"""ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], marker='x', color='red')

for i, v in enumerate(pose):
    label = f'{i}' 
    ax.text(v[0], v[1], v[2], label)
"""
#print(corners)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

max = np.max(img[:,:,0][img[:,:,0] != 0])
min = np.min(img[:,:,0])
print(max)
plt.imshow((img[:,:,1] - min) / max)
plt.show()

file = '../mat/CMU/female/Part 1/seq22-body0_point_cloud_1.mat'
mat = scipy.io.loadmat(file)
img1 = mat['img']
M = find_projection_matrix(img1)
print(M)
x = img1[:,:,0]
y = img1[:,:,1]
z = img1[:,:,2]
mask = np.logical_or(np.logical_or(x != 0, y != 0), z != 0)
points = img1[mask]    
h, w, x = img1.shape
vu = np.mgrid[0:h, 0:w]

px = vu[1][mask]
py = vu[0][mask]

file = '../mat/CMU/female/Part 1/seq22-body10_point_cloud_1.mat'
mat = scipy.io.loadmat(file)
img2 = mat['img']
M2 = find_projection_matrix(img2)

for i in range(0, len(px), 100):
    x, y, z = points[i]
    point = np.array([x, y, z, 1])
    print(px[i], py[i]) 
    proj = M @ point
    print(proj / proj[-1])
    print()

"""
print()
file = '../mat/CMU/female/Part 1/seq22-body0_point_cloud_1.mat'
mat = scipy.io.loadmat(file)
img = mat['img']
M = find_projection_matrix(img)
print(M)

file = '../mat/CMU/female/Part 1/seq22-body1_point_cloud_1.mat'
mat = scipy.io.loadmat(file)
img = mat['img']
M = find_projection_matrix(img)
print(M)
"""
"""
print()
file = '../mat/CMU/female/Part 1/seq22-body0_point_cloud_1.mat'
mat = scipy.io.loadmat(file)
img = mat['img']
M = find_projection_matrix(img)
print(M)

file = '../mat/CMU/female/Part 1/seq22-body0_point_cloud_1.mat'
mat = scipy.io.loadmat(file)
img = mat['img']
M = find_projection_matrix(img)
print(M)
file = '../mat/CMU/male/Part 4/seq606-body0_point_cloud_1.mat'
mat = scipy.io.loadmat(file)
img = mat['img']
M = find_projection_matrix(img)
print(M)

print()
file = '../mat/CMU/female/Part 1/seq22-body0_point_cloud_2.mat'
mat = scipy.io.loadmat(file)
img = mat['img']
M = find_projection_matrix(img)
print(M)

file = '../mat/CMU/male/Part 4/seq606-body0_point_cloud_2.mat'
mat = scipy.io.loadmat(file)
img = mat['img']
M = find_projection_matrix(img)
print(M)

print()
file = '../mat/CMU/female/Part 1/seq22-body0_point_cloud_3.mat'
mat = scipy.io.loadmat(file)
img = mat['img']
M = find_projection_matrix(img)
print(M)

file = '../mat/CMU/male/Part 4/seq606-body0_point_cloud_3.mat'
mat = scipy.io.loadmat(file)
img = mat['img']
M = find_projection_matrix(img)
print(M)

print()
file = '../mat/CMU/female/Part 1/seq22-body0_point_cloud_4.mat'
mat = scipy.io.loadmat(file)
img = mat['img']
M = find_projection_matrix(img)
print(M)

file = '../mat/CMU/male/Part 4/seq606-body0_point_cloud_4.mat'
mat = scipy.io.loadmat(file)
img = mat['img']
M = find_projection_matrix(img)
print(M)
"""
