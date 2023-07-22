import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pickle


def lab_to_rgb(lab):
    # Convert LAB to RGB color space
    lab = lab.reshape(-1, 3)
    xyz = np.zeros_like(lab)
    mask = lab[:, 0] > 7.9996
    xyz[mask, 1] = (lab[mask, 0] + 16.0) / 116.0
    xyz[~mask, 1] = lab[~mask, 0] / 903.3
    xyz[:, 0] = lab[:, 1] / 500.0 + xyz[:, 1]
    xyz[:, 2] = xyz[:, 1] - lab[:, 2] / 200.0
    xyz = np.where(xyz ** 3 > 0.008856, xyz, (xyz - 16.0 / 116.0) / 7.787)
    xyz = xyz * [0.950456, 1.0, 1.088754]
    m = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    rgb = np.dot(xyz, m.T)
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb.reshape(lab.shape)


def plot_ab_space(colors_bins):
    # Create a meshgrid for A and B values
    a_vals = np.linspace(-100, 100, 200)
    b_vals = np.linspace(-100, 100, 200)
    A, B = np.meshgrid(a_vals, b_vals)
    lab_grid = np.dstack((np.ones_like(A) * 50, A, B))  # Set L to 50 to have a middle gray background

    # Convert LAB to RGB
    rgb_grid = lab_to_rgb(lab_grid)

    # Flatten the grids for plotting
    A = A.flatten()
    B = B.flatten()
    rgb = rgb_grid.reshape(-1, 3)

    # Create the figure and axis
    plt.figure()
    plt.xlabel("A")
    plt.ylabel("B")

    # Plot the colors in the LAB space without edges
    plt.scatter(A, B, c=rgb, marker='o', s=15)
    plt.scatter(colors_bins[:, 0], colors_bins[:, 1], color='black', s=15)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(fr"color_bins.jpg", bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    with open('color_bins_64_prelim.pickle', 'rb') as handle:
        color_bins = pickle.load(handle)

    plot_ab_space(color_bins)



