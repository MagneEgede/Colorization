import pickle
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import cv2
from skimage import io, color
import os
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler


def kernel_function(wj, v, sigma=1):
    # Calculation of the squared euclidean distance between w_j and v.
    distance = np.sum((wj - v) ** 2)

    # Calculation of kernel value using Gaussian kernel formula.
    kernel_value = np.exp(-distance / (2 * sigma ** 2))

    return kernel_value


def calculate_lab_ssim_psnr(original_image, reconstructed_image):
    # Calculation of SSIM and PSNR using functions from skimage.metrics.
    ssim_value = ssim(original_image, reconstructed_image, channel_axis=2, data_range=1.0)
    psnr_value = psnr(original_image, reconstructed_image, data_range=1.0)

    return ssim_value, psnr_value


# Read relevant data import from training_code.py + color_bins.py.
with open('filename_a_prelim.pickle', 'rb') as handle:
    a_channel = pickle.load(handle)

with open('filename_b_prelim.pickle', 'rb') as handle:
    b_channel = pickle.load(handle)

with open('filename_descriptor_prelim.pickle', 'rb') as handle:
    descriptor = pickle.load(handle)

with open('filename_std_prelim.pickle_ext', 'rb') as handle:
    std = pickle.load(handle)

with open('filename_lap_prelim.pickle_ext', 'rb') as handle:
    lap = pickle.load(handle)

with open('filename_l_prelim.pickle', 'rb') as handle:
    l = pickle.load(handle)

with open('color_bins_64_prelim.pickle', 'rb') as handle:
    color_bins = pickle.load(handle)

with open('labels_64_prelim.pickle', 'rb') as handle:
    labels = pickle.load(handle)

# Normalized descriptors from training + stored scaling to utilize later.
descriptor = pd.DataFrame(np.concatenate(descriptor))
scaler1 = StandardScaler()
descriptor = pd.DataFrame(scaler1.fit_transform(descriptor), columns=descriptor.columns)

# Perform PCA with 10 components - 10 is chosen a trade-off between time and accuracy.
pca = PCA(n_components=10)
pca_features = pca.fit_transform(descriptor)
features = np.c_[pca_features, l, std, lap]

# Looking at PCA performance.
components = pca.components_

# Extracting the eigenvalues.
eigenvalues = pca.explained_variance_

# Printing the explained variance ratio.
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)

# Features contain a normalized version of pixel descriptors, 10 from pca, 1 from respectively l, std and lap.
scaler2 = StandardScaler()
features = scaler2.fit_transform(features)

# Create a KDTree from the data points, could look at leaf size.
kdtree = cKDTree(features)  # , leafsize=20)

# Dividing into general train set and test set.
train = np.load('trainset.npy')

# Initiating test process.
counter = 0
test_results_metrics = []
folder = 'color'

# Looping through decided test images.
for filename in os.listdir(folder):
    if filename.endswith(('jpg', 'png')):
        if train[int(filename.split('.')[0])] == 0:
            print(int(filename.split('.')[0]))
            if 50 <= counter < 100:

                # Collect gray and color version - colored version used to get gray_color (lumen).
                gray_test = io.imread(fr'gray/{filename}')
                lab_image = color.rgb2lab(io.imread(os.path.join(folder, filename)))

                gray_color, a_channel, b_channel = cv2.split(lab_image)
                gray = cv2.cvtColor(io.imread(os.path.join(folder, filename)), cv2.COLOR_RGB2GRAY)

                patch_size = 5
                descriptors = []

                # Create SIFT object.
                sift = cv2.xfeatures2d.SIFT_create()

                a_color = np.zeros_like(a_channel)
                b_color = np.zeros_like(b_channel)

                t1 = time.time()

                # Calculation of the weighted standard deviation of intensity in a 5x5 neighborhood.
                kernel_size = 5
                kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
                kernel /= kernel_size * kernel_size
                mean, squared_mean = cv2.filter2D(gray, -1, kernel), cv2.filter2D(gray ** 2, -1, kernel)
                weighted_std_dev = np.sqrt(squared_mean - mean ** 2)

                # Computation of the smooth version of the Laplacian.
                smoothed_gray = cv2.GaussianBlur(gray, (5, 5), 0)
                laplacian = cv2.Laplacian(smoothed_gray, cv2.CV_64F)

                # Looking at the amount of labels within each color bin.
                length_of_label = []
                for i in range(len(color_bins)):
                    length_of_label.append(len(labels[labels == i]))

                # Looping through pixels in gray scale image to find descriptor and
                # select color based on nearest neighbours distance from training set.
                for pixel_y in range(patch_size // 2, gray_test.shape[0] - patch_size // 2):
                    t2 = time.time()
                    print(t2 - t1)
                    print(pixel_y)
                    t1 = time.time()

                    for pixel_x in range(patch_size // 2, gray_test.shape[1] - patch_size // 2):

                        # Create a keypoint for the looped through pixel.
                        keypoint = cv2.KeyPoint(x=pixel_x, y=pixel_y, size=patch_size)

                        # Compute the SIFT descriptor for the keypoint.
                        _, descriptor = sift.compute(gray_test, [keypoint])
                        descriptor = scaler1.transform(descriptor)
                        pcaed_descriptor = pca.transform(descriptor)

                        # Query the KDTree for the 100 closest neighbors.
                        query_point = np.append(pcaed_descriptor,
                                                [gray_color[pixel_x, pixel_y], weighted_std_dev[pixel_x, pixel_y],
                                                 laplacian[pixel_x, pixel_y]])

                        k = 100  # Number of nearest neighbors to retrieve.

                        query_point = scaler2.transform(query_point.reshape(1, -1)).flat
                        distances, indices = kdtree.query(query_point, k)

                        # Calculate estimators based on Gaussian kernel for 100 closest neighbors.
                        A = [kernel_function(features[i, :], query_point) for i in
                             indices.tolist()]
                        Alabels = [labels[j] for j in indices.tolist()]

                        # Colorization of pixel based on training set and centroids in color bins.

                        weight_bin_i = []
                        indexes_list = []
                        for i in range(len(color_bins)):
                            indexes = [j for j, x in enumerate(Alabels) if (x == i)]
                            indexes_list.append(indexes)
                            if len(indexes) != 0:
                                weight_bin_i.append(sum([A[j] for j in indexes]) / sum(A))
                                # weight_bin_i.append(sum([A[j] for j in indexes]) / sum(A)  * length_of_label[i] / len(labels))
                            else:
                                weight_bin_i.append(0)

                        a_color[pixel_x, pixel_y] = color_bins[np.argmax(weight_bin_i), 0]
                        b_color[pixel_x, pixel_y] = color_bins[np.argmax(weight_bin_i), 1]

                # Transform to RGB.
                merged_lab_image = cv2.merge([gray_color, a_color, b_color])
                rgb_image = color.lab2rgb(merged_lab_image)

                # Store RGB version of recolorized solution.
                plt.imshow(rgb_image)
                plt.axis('off')
                plt.savefig(fr"Magne_{int(filename.split('.')[0])}_K100_ext.jpg", bbox_inches='tight')
                # plt.show()

                # Metrics to compare with other methods.
                mae_a = np.mean(mean_absolute_error(a_color, a_channel, multioutput='raw_values'))
                print("Mean Absolute Error a:", mae_a)

                mae_b = np.mean(mean_absolute_error(b_color, b_channel, multioutput='raw_values'))
                print("Mean Absolute Error b:", mae_b)

                mse_a = np.mean(mean_squared_error(a_color, a_channel, multioutput='raw_values', squared=False))
                print("Mean Squared Error a:", mse_a)

                mse_b = np.mean(mean_squared_error(b_color, b_channel, multioutput='raw_values', squared=False))
                print("Mean Squared Error b:", mse_b)

                ssim_value, psnr_value = calculate_lab_ssim_psnr(
                    cv2.imread(os.path.join(folder, filename)).astype(float) / 255, rgb_image)
                print("SSIM:", ssim_value)
                print("PSNR:", psnr_value)

                # Store metrics.
                test_results_metrics.append([mae_a, mae_b, mse_a, mse_b, ssim_value, psnr_value])

            counter += 1

# Dumping metrics
with open('test_results_metrics_50_100.pickle', 'wb') as handle:
    pickle.dump(test_results_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
