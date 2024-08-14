from matplotlib import pyplot as plt


class Sketch:

    @staticmethod
    def initialise_axes(figure):
        ax = figure.add_subplot(projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
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
    def plot_annotated_ptcloud(pt_cloud, annotation, show=True, ticks=False):
        colors = ['pink', 'magenta', 'red', 'crimson', 'brown', 'orange',
                  'beige', 'yellow', 'green', 'lime', 'darkcyan', 'blue',
                  'aqua', 'navy', 'purple', 'violet', 'gray' 'black']
        annotations = {}
        for i in range(len(pt_cloud)):
            key = annotation[i]
            point = pt_cloud[i]

            if key not in annotations:
                annotations[key] = [point]
            else:
                annotations[key].append(point)

        fig = plt.figure()
        ax = Sketch.initialise_axes(fig)
        if not ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

        for key in annotations.keys():
            points = annotations[key]
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z = [p[2] for p in points]
            ax.scatter(x, y, z, marker=".", c=colors[key % 18])

        if show:
            plt.show()

        return fig
