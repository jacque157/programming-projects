import numpy as np
from matplotlib import pyplot as plt
from  matplotlib import animation as animation


def plot_depth_map(depth_map):
    plt.imshow(np.abs(depth_map),  cmap='gray')
    
def plot_point_cloud(point_cloud, ax=None, step=10):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        
    point_cloud = np.random.permutation(point_cloud)[::step, :]
    ax.scatter(point_cloud[:, 0], 
               point_cloud[:, 1], 
               point_cloud[:, 2], 
               marker='o', 
               color='blue', 
               alpha=0.1)
    return ax

def plot_structured_point_cloud(structured_point_cloud, ax=None, step=10):
    xs = structured_point_cloud[:, :, 0]
    ys = structured_point_cloud[:, :, 1]
    zs = structured_point_cloud[:, :, 2]
    mask = np.logical_not(np.logical_and(np.logical_and(xs == 0, ys == 0), zs == 0))
    pt_cloud = structured_point_cloud[mask]
    return plot_point_cloud(pt_cloud, ax, step)

def plot_skeleton(skeleton, ax=None, plot_hands=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

    pairs = [(0, 2), (2, 5), (5, 8), (8, 11), # right leg
             (0, 1), (1, 4), (4, 7), (7, 10), # left leg
             (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), # spine
             (9, 14), (14, 17), (17, 19), (19, 21), # right arm
             (9, 13), (13, 16), (16, 18), (18, 20)] # left arm
    hands = [(21, 49), (49, 50), (50, 51),
             (21, 37), (37, 38), (38, 39),
             (21, 40), (40, 41), (41, 42),
             (21, 46), (46, 47), (47, 48),
             (21, 43), (43, 44), (44, 45),
             
             (20, 34), (34, 35), (35, 36),
             (20, 22), (22, 23), (23, 24),
             (20, 25), (25, 26), (26, 27),
             (20, 31), (31, 32), (32, 33),
             (20, 28), (28, 29), (29, 30)]
    
    for (start, end) in pairs:
        x_start, y_start, z_start = skeleton[start, :]
        x_end, y_end, z_end = skeleton[end, :]
        ax.plot((x_start, x_end),
                (y_start, y_end),
                (z_start, z_end),
                marker='x', 
                color='red')
        
    for i, v in enumerate(skeleton[:22]):
        label = f'{i}' 
        ax.text(v[0], v[1], v[2], label)

    if plot_hands and len(skeleton) >= 52:
        for (start, end) in hands:
            x_start, y_start, z_start = skeleton[start, :]
            x_end, y_end, z_end = skeleton[end, :]
            ax.plot((x_start, x_end),
                    (y_start, y_end),
                    (z_start, z_end),
                    marker='x', 
                    color='red')
            
        for i, v in enumerate(skeleton[22:]):
            label = f'{i}' 
            ax.text(v[0], v[1], v[2], label)
    return ax


def animate_silhouettes(silhouettes):
    fig, ax = plt.subplots()
    ims = []
    for i, silhouette in enumerate(silhouettes):
        im = ax.imshow(silhouette, cmap='gray', animated=True)
        if i == 0:
            ax.imshow(silhouette, cmap='gray')  # show an initial one first
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    writergif = animation.PillowWriter(fps=5)
    ani.save('silhouettes.gif', writer=writergif)
    plt.show()

def animate_depthmaps(depthmaps):
    fig, ax = plt.subplots()
    ims = []
    for i, depthmap in enumerate(depthmaps):
        im = ax.imshow(depthmap, cmap='gray', animated=True)
        if i == 0:
            ax.imshow(depthmap, cmap='gray')  # show an initial one first
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    writergif = animation.PillowWriter(fps=5)
    ani.save('depthmap.gif', writer=writergif)
    plt.show()
