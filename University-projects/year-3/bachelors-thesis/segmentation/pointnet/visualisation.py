from matplotlib import pyplot as plt
import numpy as np


class Sketch:
    @staticmethod
    def initialise_axes(figure, limits=None):
        ax = figure.add_subplot(projection='3d')
        ax.view_init(elev=0., azim=-180)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        if limits:
            ax.set_xlim(limits[0][0], limits[0][1])
            ax.set_ylim(limits[1][0], limits[1][1])
            ax.set_zlim(limits[2][0], limits[2][1])

        return ax

    @staticmethod
    def plot_points(sequence):
        fig = plt.figure()
        ax = Sketch.initialise_axes(fig)
        x = sequence[..., 0]
        y = sequence[..., 1]
        z = sequence[..., 2]
        ax.scatter(x, y, z, marker=".")
        plt.show()

    @staticmethod
    def plot_ptcloud(pt_cloud):
        Sketch.plot_points(pt_cloud)

    @staticmethod
    def plot_skeleton(pose):
        Sketch.plot_points(pose)

    @staticmethod
    def plot_annotated_ptcloud(pt_cloud, annotation, show=True, ticks=False, limits=None):
        colors = ['pink', 'magenta', 'red', 'crimson', 'brown', 'orange',
                  'beige', 'yellow', 'green', 'lime', 'darkcyan', 'blue',
                  'aqua', 'navy', 'purple', 'violet', 'gray', 'black']

        fig = plt.figure()
        ax = Sketch.initialise_axes(fig, limits)
        if not ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        for key_point in np.unique(annotation):
            region = pt_cloud[annotation == key_point]
            x = region[:, 0]
            y = region[:, 1]
            z = region[:, 2]

            ax.scatter(x, y, z, marker=".", c=colors[key_point % len(colors)])

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_difference(pt_cloud, ground_truth, prediction, show=True, ticks=False):
        blue_colour = 11
        red_colour = 2
        gray_colour = 16
        annotation = prediction.copy() # np.ones(pt_cloud.shape[0], int) * blue_colour
        annotation[prediction == ground_truth] = gray_colour #[prediction != ground_truth] = red_colour
        return Sketch.plot_annotated_ptcloud(pt_cloud, annotation, show, ticks)