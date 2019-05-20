import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform

from sklearn.cluster import KMeans
from datasets import two_moon_dataset

plt.ion()


def similarity_function(x, y):
    weights = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    num_value = np.sum(np.square(x - y) * weights)
    temperature = 0.06
    return np.exp(-num_value / temperature)


def spectral_clustering(data, num_clusteres, sigma=1.):
    num_samples, dim = data.shape
    # labels = np.random.choice(range(0, num_clustering), size=num_samples);

    affinity_matrix = np.zeros((num_samples, num_samples))
    for i in range(0, num_samples):
        print('{0} out of {1}'.format(i, num_samples))
        for j in range(i + 1, num_samples):
            affinity_matrix[i, j] = similarity_function(data[i], data[j])

    # fix the matrix
    affinity_matrix = affinity_matrix + np.transpose(affinity_matrix)

    # degree matrix
    D = np.diag(np.sum(affinity_matrix, axis=1))

    # DO LAPLACIAN
    L = D - affinity_matrix

    # compute eigenvalues and vectors
    eigv, eigV = np.linalg.eig(L)

    labels = np.zeros(num_samples)
    labels[eigV[:, 1] > 0] = 1

    num_cluster = num_clusteres

    # use Kmeans
    labels = KMeans(num_cluster).fit((eigV[:, 0:num_cluster])).labels_
    return labels


def main_spectral_clustering():
    # generate the dataset
    data, cl = two_moon_dataset(num_samples=300, noise=0.1)

    # visualize the datase
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)

    # run che spectral clustering
    labels = spectral_clustering(data, num_clusteres=2, sigma=0.1)

    # visualize results
    ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)
    plt.waitforbuttonpress()


def main_spectral_clustering_image():
    num_cluster = 4

    # generate the dataset
    img = skimage.io.imread('./img/soccer.jpg')
    img = skimage.transform.rescale(img, 0.03, preserve_range=True)
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
    labels = spectral_clustering(colors, num_clusteres=num_cluster, sigma=0.1)

    # visualize result
    img_labels = np.reshape(np.float32(labels) * (255.0 / num_cluster), (w, h))

    ax[1].imshow(np.uint8(img_labels))
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main_spectral_clustering_image()
