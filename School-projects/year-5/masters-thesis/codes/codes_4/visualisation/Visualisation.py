import numpy as np
from matplotlib import pyplot as plt
from  matplotlib import animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

def plot_skeleton(skeleton, ax=None, plot_hands=False, plot_labels=True, color='red', marker='x', alpha=1.0):
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
                marker=marker, 
                color=color,
                alpha=alpha)

    if plot_labels:
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
                    marker=marker, 
                    color=color,
                    alpha=alpha)

        if plot_labels:
            for i, v in enumerate(skeleton[22:]):
                label = f'{i}' 
                ax.text(v[0], v[1], v[2], label)
    return ax

def plot_example(point_cloud, predicted_skeleton=None, target_skeleton=None, plot_hands=False, plot_labels=False):
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(projection='3d')
    ax.set_axis_off()
    #ax.set_xlabel('X Axis')
    #ax.set_ylabel('Y Axis')
    #ax.set_zlabel('Z Axis')
    alpha = np.maximum(np.minimum(0.1 * np.exp((len(point_cloud) / -2048) + 1), 1.0), 1e-2)
    ax.scatter(point_cloud[:, 0], 
                   point_cloud[:, 1], 
                   point_cloud[:, 2], 
                   marker='o', 
                   color='blue', 
                   alpha=alpha)
    """if len(point_cloud) <= 2048:
        ax.scatter(point_cloud[:, 0], 
                   point_cloud[:, 1], 
                   point_cloud[:, 2], 
                   marker='o', 
                   color='blue', 
                   alpha=0.1)
    else:
        ax.scatter(point_cloud[:, 0], 
                   point_cloud[:, 1], 
                   point_cloud[:, 2], 
                   marker='o', 
                   color='blue', 
                   alpha=0.01)"""
    if predicted_skeleton is not None:
        ax = plot_skeleton(predicted_skeleton, ax=ax, plot_hands=plot_hands, plot_labels=plot_labels, color='red', marker='', alpha=1.0)
    if target_skeleton is not None:
        ax = plot_skeleton(target_skeleton, ax=ax, plot_hands=plot_hands, plot_labels=plot_labels, color='blue', marker='', alpha=0.4)
    ax.view_init(elev=10., azim=136.)
    plt.show()

def plot_dataset_examples(point_clouds, skeletons, columns):
    rows = len(point_clouds) // columns
    cols = columns
    fig = plt.figure()#figsize=(12,9))
    
    for i, (point_cloud, skeleton) in enumerate(zip(point_clouds, skeletons)):
        row = i // rows
        col = i % rows

        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax.set_axis_off()
        alpha = np.maximum(np.minimum(0.1 * np.exp((len(point_cloud) / -2048) + 1), 1.0), 1e-2)
        ax.scatter(point_cloud[:, 0], 
                   point_cloud[:, 1], 
                   point_cloud[:, 2],
                   s=0.5,
                   marker='o', 
                   color='blue', 
                   alpha=alpha)
        ax = plot_skeleton(skeleton, ax=ax, plot_hands=True, plot_labels=False, color='blue', marker='', alpha=0.4)
        ax.view_init(elev=10., azim=136.)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.show()

def animate_predictions(point_clouds, gt_skeletons, predicted_skeletons):  
    for i, (pt_cloud, gt, pred) in enumerate(zip(point_clouds, gt_skeletons, predicted_skeletons)):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        alpha = np.maximum(np.minimum(0.1 * np.exp((len(point_clouds) / -2048) + 1), 1.0), 1e-2)
        ax.set_axis_off()
        ax.axes.set_xlim3d(left=-500, right=500) 
        ax.axes.set_ylim3d(bottom=-500, top=500) 
        ax.axes.set_zlim3d(bottom=-100, top=1500)
        
        ax.scatter(pt_cloud[:, 0], 
                       pt_cloud[:, 1], 
                       pt_cloud[:, 2],
                       s=0.9,
                       marker='o', 
                       color='blue', 
                       alpha=alpha)
        ax = plot_skeleton(pred, ax=ax, plot_hands=False, plot_labels=False, color='red', marker='', alpha=1.0)
        ax = plot_skeleton(gt, ax=ax, plot_hands=False, plot_labels=False, color='blue', marker='', alpha=0.4)
        ax.view_init(elev=10., azim=136.)
        plt.savefig(f'anims/prediction_frame_{i}.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ims = []
    imgs = [mpimg.imread(f'anims/prediction_frame_{i}.png') for i in range(len(point_clouds))]
    for i, img in enumerate(imgs):
        im = ax.imshow(img, animated=True)
        if i == 0:
            ax.imshow(img, cmap='gray')  # show an initial one first
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
    writergif = animation.PillowWriter(fps=2)
    ani.save('anims/predictions2.gif', writer=writergif)

def animate_silhouettes(silhouettes):
    fig, ax = plt.subplots()
    ims = []
    for i, silhouette in enumerate(silhouettes):
        im = ax.imshow(silhouette, cmap='gray', animated=True)
        if i == 0:
            ax.imshow(silhouette, cmap='gray')  # show an initial one first
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    writergif = animation.PillowWriter(fps=2)
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
    writergif = animation.PillowWriter(fps=2)
    ani.save('depthmap.gif', writer=writergif)
    plt.show()

def animate_images(images):
    fig, ax = plt.subplots()
    ims = []
    for i, img in enumerate(images / 255):
        im = ax.imshow(img, animated=True)
        if i == 0:
            ax.imshow(img)  # show an initial one first
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=1000)
    writergif = animation.PillowWriter(fps=2)
    ani.save('images.gif', writer=writergif)
    plt.show()

def plot_MPJPEs(per_sample_MPJPE, splits=100, min_=None, max_=None):
    if min_ is None:
        min_ = np.min(per_sample_MPJPE)
    if max_ is None:
        max_ = np.max(per_sample_MPJPE)
    d = max_ - min_
    tresholds = [max_ - (d * i / splits) for i in range(splits + 1)]
    counts = [100 * np.sum(per_sample_MPJPE <= treshold) / len(per_sample_MPJPE) for treshold in tresholds]
    plt.plot(tresholds, counts)
    plt.xlabel('MPJPE [mm]')

    end = int(max_) 
    ticks = [i for i in range(0, end + 1, 20)]
    plt.xticks(ticks, rotation=45)

    plt.ylabel('Percentage ')
    plt.yticks(range(0, 101, 5))
    
    plt.xlim([0, max_])
    plt.ylim([0, 101])
    plt.grid(True)
    plt.show()
    
def plot_MPJPEs_histogram(per_sample_MPJPE, splits=100):
    per_sample_MPJPE.sort()
    per_sample_MPJPE = np.array(per_sample_MPJPE)
    tresholds = [per_sample_MPJPE[int(len(per_sample_MPJPE) * i / splits)] for i in range(splits)]
    #tresholds.append(tresholds[-1])
    percentage = [100 * np.sum(per_sample_MPJPE <= treshold) / len(per_sample_MPJPE) for treshold in tresholds]
    a = np.arange(len(tresholds))
    plt.bar(a, percentage)
    #tresholds = [str(t) for t in tresholds]
    plt.xlabel('MPJPE [mm]')
    plt.xticks(a, [round(t, 3) for t in tresholds])
    
    plt.xticks(rotation=45)
    plt.ylabel('Percentage')
    plt.yticks(np.arange(0, 100+1, 5))
    plt.grid(True, axis='y')
    plt.show()

def plot_MPJPEs_histogram_2(per_sample_MPJPE, parts=5, min_=None, max_=None):
    per_sample_MPJPE.sort()
    per_sample_MPJPE = np.array(per_sample_MPJPE)

    if min_ is None:
        min_ = per_sample_MPJPE[0]
    if max_ is None:
        max_ = per_sample_MPJPE[-1]
    
    start = ((int(min_) // parts) + 1) * parts
    end = (int(max_) // parts) * parts

    tresholds = range(start, end + 1, parts)
    percentage = [100 * np.sum(per_sample_MPJPE <= treshold) / len(per_sample_MPJPE) for treshold in tresholds]
    a = np.arange(len(tresholds))
    plt.bar(a, percentage)
    #tresholds = [str(t) for t in tresholds]
    plt.xlabel('MPJPE [mm]')
    plt.xticks(a, [round(t, 3) for t in tresholds])
    
    plt.xticks(rotation=45)
    plt.ylabel('Percentage')
    plt.yticks(np.arange(0, 100+1, 5))
    plt.grid(True, axis='y')
    plt.show()


def plot_per_joint_error(per_sample_per_join_MPJPE):
    s, j = per_sample_per_join_MPJPE.shape
    per_sample_per_join_MPJPE = np.array(per_sample_per_join_MPJPE)
    per_joint_error = np.mean(per_sample_per_join_MPJPE, 0)
    per_joint_std = np.std(per_sample_per_join_MPJPE, 0)
    mpjpe = np.mean(per_sample_per_join_MPJPE)
    mpjpe_std = np.std(per_sample_per_join_MPJPE)

    errors = np.concatenate((per_joint_error, [mpjpe]))
    stds = np.concatenate((per_joint_std, [mpjpe_std]))
    
    a = np.arange(len(errors))
    b = [i + 0.5 for i in a]
    plt.bar(a, errors, width=0.5, 
        color='navy', label ='Average') 
    plt.bar(b, stds, width=0.5, 
        color='violet', label ='Standard Deviation') 
    #tresholds = [str(t) for t in tresholds]
    plt.xticks([i + 0.25 for i in a], 
        [f'Joint: {i}' for i in range(j)] + ['Mean'])
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.yticks(np.arange(0, 160+1, 10))
    plt.ylabel('Error [mm]')  
    plt.legend()
    
    plt.show()
