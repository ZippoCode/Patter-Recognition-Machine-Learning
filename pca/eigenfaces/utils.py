"""
Some plotting functions
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

plt.ion()


def show_eigenfaces(eigenfaces, size):
    """
    Plots ghostly pca

    Parameters
    ---------------------------------
    :param eigenfaces: ndarray
            pca (eigenvectors of face covariance matrix)
    :param size: tuple
            the size of each face image like (h,w)
    Returns
    ----------------------------------
    :return: None
    """

    eigf = []

    for f in eigenfaces.T.copy():
        f -= f.min()
        f /= f.max() + np.finfo(float).eps

        eigf.append(np.reshape(f, newshape=size))

        to_show = np.concatenate(eigf, axis=1)

        plt.imshow(to_show)
        plt.title("Eigenfaces")
        plt.waitforbuttonpress()


def show_3d_faces_with_class(points, labels):
    """
    Plots 3d data in colorful point (color is class)

    Parameters
    ---------------------------------
    :param points: ndarray
            3d points to plot (shape: (n_samples, 3)
    :param labels: ndarray
            classes (shape: (n_samples))

    Returns
    -----------------------------------
    :return: None
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 1], alpha=0.5, c=labels, s=60)
    plt.show(block=True)
