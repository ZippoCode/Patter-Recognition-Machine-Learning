import numpy as np
from numpy.matlib import repmat

import matplotlib.pyplot as plt

from utils.datasets import gaussians_dataset

import cv2

plt.ion()


def kmeans(data, n_cl, verbose=True):
    """
    Kmeans algorithm.

    Parameters
    ----------
    data: ndarray
        data to partition, has shape (n_samples, dimensionality).
    n_cl: int
        number of clusters.
    verbose: bool
        whether or not to plot assignment at each iteration (default is True).

    Returns
    -------
    ndarray
        computed assignment. Has shape (n_samples,)
    """
    n_samples, dim = data.shape

    # initialize centers
    centers = data[np.random.choice(range(n_samples), size=n_cl)]

    old_labels = np.zeros(shape=n_samples)
    while True:  # stopping criterion

        # assign
        distances = np.zeros(shape=(n_samples, n_cl))
        for c_idx, c in enumerate(centers):
            square = np.square(data - repmat(c, n_samples, 1))
            distances[:, c_idx] = np.sum(square, axis=1)

        new_labels = np.argmin(distances, axis=1)

        # re-estimate
        for l in range(0, n_cl):
            centers[l] = np.mean(data[new_labels == l], axis=0)

        if np.all(new_labels == old_labels):
            break

        # update
        old_labels = new_labels

    return new_labels


def main_kmeans_gaussian(n_cl=3):
    """
    Main function to run kmeans the synthetic gaussian dataset.

    :param:
        - n_cl: number of clusters (default is three)
    """

    # generate the dataset
    data, cl = gaussians_dataset(3, [100, 100, 70], [[1, 1], [-4, 6], [8, 8]], [[1, 1], [3, 3], [1, 1]])

    # visualize the dataset
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)

    # solve kmeans optimization
    labels = kmeans(data, n_cl=n_cl, verbose=False)

    # visualize results
    ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)
    plt.waitforbuttonpress()


def main_kmeans_img(img_path):
    """
    Main function to run kmeans for image segmentation.

    :parameter
    ----------
    img_path: str
        Path of the image to load and segment.

    :return:
    -------
    None
    """

    # load the image
    img = np.float32(cv2.imread(img_path))
    h, w, c = img.shape

    # visualize image
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8))
    ax[0].axis('off')

    # add coordinates
    row_indexes = np.arange(0, h)
    col_indexes = np.arange(0, w)
    coordinates = np.zeros(shape=(h, w, 2))
    coordinates[..., 0] = repmat(row_indexes, w, 1).T
    coordinates[..., 1] = repmat(col_indexes, h, 1)

    data = np.concatenate((img, coordinates), axis=-1)
    data = np.reshape(data, newshape=(w * h, 5))

    # solve K-means optimization
    labels = kmeans(data, n_cl=2, verbose=False)
    ax[1].imshow(np.reshape(labels, (h, w)), cmap='hot')
    ax[1].axis('off')
    plt.waitforbuttonpress()


if __name__ == '__main__':
    # main_kmeans_img('emma.png')
    main_kmeans_gaussian(5)
