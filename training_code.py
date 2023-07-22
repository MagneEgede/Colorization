import cv2
import os
import pickle
from skimage import io, color
import numpy as np


def compute_complete_descriptor(folder, patch_size=16):
    descriptors = []
    a_chan = []
    b_chan = []
    L_chan = []
    lap = []
    std = []

    # Dividing into general train set and test set.
    train = np.load('trainset.npy')

    # Looping through decided train images.
    for filename in os.listdir(folder):
        if filename.endswith(('jpg', 'png')):
            if train[int(filename.split('.')[0])] == 1:
                print(int(filename.split('.')[0]))

                # Collect gray and color version - colored version used to get gray_color (lumen).
                image = io.imread(os.path.join(folder, filename))

                lab_image = color.rgb2lab(image)

                gray_color, a_channel, b_channel = cv2.split(lab_image)
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

                # Calculation of the weighted standard deviation of intensity in a 5x5 neighborhood.
                kernel_size = 5
                kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
                kernel /= kernel_size * kernel_size
                mean, squared_mean = cv2.filter2D(gray, -1, kernel), cv2.filter2D(gray ** 2, -1, kernel)
                weighted_std_dev = np.sqrt(squared_mean - mean ** 2)

                # Computation of the smooth version of the Laplacian.
                smoothed_gray = cv2.GaussianBlur(gray, (5, 5), 0)
                laplacian = cv2.Laplacian(smoothed_gray, cv2.CV_64F)

                # Create SIFT object
                sift = cv2.xfeatures2d.SIFT_create()  # nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)

                for pixel_y in range(patch_size // 2, image.shape[0] - patch_size // 2):
                    for pixel_x in range(patch_size // 2, image.shape[1] - patch_size // 2):

                        # Create a keypoint for the looped through pixel.
                        keypoint = cv2.KeyPoint(x=pixel_x, y=pixel_y, size=patch_size)

                        # Compute the SIFT descriptor for the keypoint.
                        _, descriptor = sift.compute(gray, [keypoint])

                        # Appending relevant color and local descriptors.
                        descriptors.append(descriptor)
                        a_chan.append(a_channel[pixel_x, pixel_y])
                        b_chan.append(b_channel[pixel_x, pixel_y])
                        L_chan.append(gray_color[pixel_x, pixel_y])
                        std.append(weighted_std_dev[pixel_x, pixel_y])
                        lap.append(laplacian[pixel_x, pixel_y])

                print('pic done')

    return descriptors, a_chan, b_chan, L_chan, std, lap


# patch_size for SIFT, another hyperparameter that could be changed. Neighbourhood size of SIFT calculations.
patch_size = 5

# Obtain local descriptors of train images with 13 features each for every pixel.
descriptor, a_chan, b_chan, L_chan, std, lap = compute_complete_descriptor('sub_color', patch_size)


# Dump all needed results from training set.
with open('filename_descriptor_prelim.pickle', 'wb') as handle:
    pickle.dump(descriptor, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename_a_prelim.pickle', 'wb') as handle:
    pickle.dump(a_chan, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename_b_prelim.pickle', 'wb') as handle:
    pickle.dump(b_chan, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename_l_prelim.pickle', 'wb') as handle:
    pickle.dump(L_chan, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename_std_prelim.pickle_ext', 'wb') as handle:
    pickle.dump(std, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename_lap_prelim.pickle_ext', 'wb') as handle:
    pickle.dump(lap, handle, protocol=pickle.HIGHEST_PROTOCOL)
