import pickle
import numpy as np
import cv2
from skimage import io, color
import matplotlib.pyplot as plt

# This script is validation of color bins and if the amount is sufficient to be a good approximation of images.
# 64 - 128 bins seem sufficient.

with open('color_bins_prelim.pickle', 'rb') as handle:
    color_bins = pickle.load(handle)

with open('labels_prelim.pickle', 'rb') as handle:
    labels = pickle.load(handle)

with open('filename_a_prelim.pickle_ext', 'rb') as handle:
    a_channel_check = np.array(pickle.load(handle))

with open('filename_b_prelim.pickle_ext', 'rb') as handle:
    b_channel_check = np.array(pickle.load(handle))

lab_pic_check = labels[:146 * 146]
a_channel_check = np.array(a_channel_check[:146 * 146]).reshape((146, 146)).T
b_channel_check = np.array(b_channel_check[:146 * 146]).reshape((146, 146)).T

a_pic_check = np.array([color_bins[i, 0] for x, i in enumerate(lab_pic_check)])
b_pic_check = np.array([color_bins[i, 1] for x, i in enumerate(lab_pic_check)])

lab_pic_1 = np.array(lab_pic_check).reshape((146, 146)).T
a_pic_1 = np.array(a_pic_check).reshape((146, 146)).T
b_pic_1 = np.array(b_pic_check).reshape((146, 146)).T

gray_test = io.imread('gray/0.jpg')
lab_image = color.rgb2lab(io.imread('color/0.jpg'))

gray_color, a_channel, b_channel = cv2.split(lab_image)
a_channel = a_channel[2:148, 2:148]
b_channel = b_channel[2:148, 2:148]
gray_color = gray_color[2:148, 2:148]

plt.figure()
merged_lab_image = cv2.merge([gray_color, a_pic_1, b_pic_1])
rgb_image = color.lab2rgb(merged_lab_image)

plt.imshow(rgb_image)
plt.axis('off')

plt.figure()
merged_lab_image = cv2.merge([gray_color, a_channel, b_channel])
rgb_image = color.lab2rgb(merged_lab_image)

plt.imshow(rgb_image)
plt.axis('off')
plt.show()