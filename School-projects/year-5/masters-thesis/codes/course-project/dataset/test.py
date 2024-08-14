import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d_x = signal.windows.gaussian(kernlen, std=std)[:, None] + 2
    gkern1d_y = signal.windows.gaussian(kernlen, std=std)[:, None] + 2
    gkern2d = np.outer(gkern1d_x, gkern1d_y)
    return gkern2d

"""plt.imshow(gkern(21), interpolation='none')
plt.show()

N = 7   # kernel size
k1d = signal.windows.gaussian(N, std=1).reshape(N, 1)
kernel = np.outer(k1d, k1d)
plt.imshow(kernel)
plt.show()

A = np.zeros((16, 16))
A[5, 9] = 1    # random
plt.imshow(A)
plt.show()

row, col = np.where(A == 1)
A[row[0]-(N//2):row[0]+(N//2)+1, col[0]-(N//2):col[0]+(N//2)+1] = kernel
plt.imshow(A)
plt.show()"""

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
    return heat_map[kernel_size : h - kernel_size, kernel_size : w - kernel_size]

heat = generate_heatmap(129, 97, 97, 129, 2, 11)
#heat /= heat.max()
print(heat.max())
print(np.sum(heat == 1))
plt.imshow(heat)
plt.show()
