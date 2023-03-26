import time
import numpy as np
import cv2
from sklearn.cluster import KMeans


def extract_seed_pixels(seed_img, rgb_img):
    # Extracting channels from seed
    blue_ch = seed_img[:, :, 0]  # Blue channel in BGR
    red_ch = seed_img[:, :, 2]  # Red channel in BGR

    # Extracting masks from rgb image
    background_mask = cv2.bitwise_and(rgb_img, rgb_img, mask=blue_ch)
    foreground_mask = cv2.bitwise_and(rgb_img, rgb_img, mask=red_ch)

    # Extracting only the colored pixels to reduce computational time
    non_black_pixels = background_mask.any(axis=-1)
    background = background_mask[non_black_pixels]
    non_black_pixels = foreground_mask.any(axis=-1)
    foreground = foreground_mask[non_black_pixels]

    return background, foreground


def kmeans(k, data, use_builtin=False):
    # Use builtin function
    if use_builtin:
        km = KMeans(n_clusters=k)
        km.fit(data)
        return km.labels_, km.cluster_centers_

    # Use custom implementation
    else:
        # Initialize centroids randomly
        centroids = data[np.random.choice(data.shape[0], k, replace=False), :]

        while True:
            # Assign each data point to the nearest centroid
            distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
            cluster_idx = np.argmin(distances, axis=1)

            # Update centroids as the mean of the data points assigned to each cluster
            new_centroids = np.array([data[cluster_idx == i, :].mean(axis=0)
                                      if np.sum(cluster_idx == i) > 0 else centroids[i]
                                      for i in range(k)])

            # Check if centroids have moved
            if np.allclose(new_centroids, centroids):
                break

            # Updating current centroid to new for next iteration
            centroids = new_centroids

        return cluster_idx, centroids


def compute_likelihood(rgb_img, bg_cluster_idx, bg_centroids, fg_cluster_idx, fg_centroids):
    # Calculating weights
    bg_weights = np.bincount(bg_cluster_idx, minlength=len(bg_centroids)) / len(bg_cluster_idx)
    fg_weights = np.bincount(fg_cluster_idx, minlength=len(fg_centroids)) / len(fg_cluster_idx)

    # Compute distance and exponential function
    bg_dist = np.linalg.norm(rgb_img[:, :, np.newaxis, :] - bg_centroids, axis=-1)
    bg_expo = np.exp(-bg_dist)**2 * bg_weights
    fg_dist = np.linalg.norm(rgb_img[:, :, np.newaxis, :] - fg_centroids, axis=-1)
    fg_expo = np.exp(-fg_dist)**2 * fg_weights

    # Compute probabilities for each pixel and centroid
    bg_prob = np.sum(bg_expo, axis=2)
    fg_prob = np.sum(fg_expo, axis=2)

    # Set foreground pixels to white and background pixels to black
    bw = np.zeros_like(rgb_img)
    bw[fg_prob > bg_prob] = 255

    return bw


def interactive_foreground_segmentation(seed_img, rgb_img, k):
    # Extracting the seed pixels
    t = time.time()
    bg, fg = extract_seed_pixels(seed_img, rgb_img)

    # Using k-means to compute clusters for both classes
    bg_cluster_idx, bg_centroids = kmeans(k, bg, use_builtin=True)
    fg_cluster_idx, fg_centroids = kmeans(k, fg, use_builtin=True)

    # Computing likelihood
    bw = compute_likelihood(rgb_img, bg_cluster_idx, bg_centroids, fg_cluster_idx, fg_centroids)
    time_taken = time.time() - t

    # Writing results
    print("Comp Time: {:.3f}".format(time_taken), "s")
    cv2.imwrite('mask.png', bw)


k = 64
img = cv2.imread('./pics/Mona-lisa.PNG')
seed = cv2.imread('./pics/Mona-lisa stroke 1.png')
interactive_foreground_segmentation(seed, img, k)
