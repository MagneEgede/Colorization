import pickle
import numpy as np
from sklearn.cluster import KMeans


# Creating color bins to provide discrete probability density function for colors.
def create_color_bins(a_values, b_values, num_bins):
    # Stack a channel and b channel values together.
    ab_values = np.vstack((a_values, b_values)).T

    # Perform K-means clustering on training pixels.
    kmeans = KMeans(n_clusters=num_bins, random_state=42)
    kmeans.fit(ab_values)

    # Get the cluster centroids.
    color_bins = kmeans.cluster_centers_

    return color_bins, kmeans.labels_


# Train pixels' a channel and b channel
with open('filename_a_prelim.pickle', 'rb') as handle:
    a_channel = pickle.load(handle)

with open('filename_b_prelim.pickle', 'rb') as handle:
    b_channel = pickle.load(handle)

num_bins = 64  # Number of color bins

color_bins, labels = create_color_bins(a_channel, b_channel, num_bins)

# Stor number of bins and labels for training set.
with open('color_bins_64_prelim.pickle', 'wb') as handle:
    pickle.dump(color_bins, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('labels_64_prelim.pickle', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
