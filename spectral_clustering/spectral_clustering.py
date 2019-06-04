import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform

from sklearn.cluster import KMeans
from utils.datasets import two_moon_dataset

plt.ion()


def similarity_function(x, y, sig=0.1, sig2=0.1):

    #weights = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    weights = np.array([1, 1], dtype=np.float32)
    num_value = np.sum(np.square(x - y) * weights)
    temperature = 0.06
    return np.exp(-num_value / temperature)

def laplacian(A):
    """
        Computes the symetric normalized laplacian

    :param A:

    :return:
        - L (shape n*n) definites as L =
    """
    num_samples, dim = A.shape

    W = np.zeros((num_samples, num_samples))
    for i in range(0, num_samples):
        print('{0} out of {1}'.format(i, num_samples))
        for j in range(i + 1, num_samples):
            W[i, j] = similarity_function(A[i], A[j])

    # fix the matrix
    W = W + np.transpose(W)

    # degree matrix
    D = np.diag(np.sum(W, axis=1))

    # DO LAPLACIAN
    L = D - W
    return L


def spectral_clustering(data, num_cluster, sigma=1.):
    L = laplacian(data)

    # compute eigenvalues and vectors
    eigv, eigV = np.linalg.eig(L)

    labels = np.zeros(data.shape[0])
    labels[eigV[:, 1] > 0] = 1

    # use Kmeans
    labels = KMeans(num_cluster).fit((eigV[:, 0:num_cluster])).labels_
    return labels


def main_spectral_clustering(num_clusteres=2):
    """
        Main function for spectral clustering.
    :param:
        - num_clusteres : int
            The number of cluster. (Default is two)
    """
    # Generate the dataset
    data, cl = two_moon_dataset(n_samples=300, noise=0.1)

    # Visualize the dataset
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)

    # Run che Spectral Clustering
    labels = spectral_clustering(data, num_clusteres, sigma=0.1)

    # Visualize results
    ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)
    plt.waitforbuttonpress()


def main_spectral_clustering_image():
    """
        Main function for Spectral Clustering
    """
    num_cluster = 2

    # generate the dataset
    img = skimage.io.imread('./img/soccer.jpg')
    img = skimage.transform.rescale(img, 0.09, preserve_range=True)
    w, h, c = img.shape

    # add i,j coordinate
    colors = np.zeros((w * h, 5))
    count = 0
    for i in range(0, w):
        for j in range(0, h):
            colors[count, :] = np.hstack(((img[i, j] / 255.0), np.float(i) / w, np.float(j) / h))
            count += 1

    # visualize the dataset
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.uint8(img))

    # run che spectral clustering
    labels = spectral_clustering(colors, num_cluster, sigma=0.1)

    # visualize result
    img_labels = np.reshape(np.float32(labels) * (255.0 / num_cluster), (w, h))

    ax[1].imshow(np.uint8(img_labels))
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main_spectral_clustering()
    # main_spectral_clustering_image()
