"""
    Function to load data from file
"""
import numpy as np
from os.path import join, basename, isdir
from glob import glob

import skimage.io as io


def get_faces_dataset(path, train_split=60):
    """
    Loads Olivelli dataset from files

    Parameters
    ------------------------------
    :param path: str
            the root folder of the Olivetti dataset
    :param train_split: int
            the percentage of dataset used for training (defaut is 60%)
    Returns
    -----------
    :return: tuple
            a tuple like (X_train, Y_train, X_test, Y_test)
    """
    cluster_folders = sorted([basename(f) for f in glob(join(path, '*')) if isdir(f)])

    Y = []
    X = []

    for cluster, cluster_f in enumerate(cluster_folders):
        img_list = glob(join(path, cluster_f, '*.pgm'))

        for i, img_path in enumerate(img_list):
            X.append(io.imread(img_path, as_grey=True).ravel())
            Y.append(cluster)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    num_samples = Y.size
    num_train_samples = (num_samples * train_split) // 100

    # shuffle
    tot = np.concatenate((X, np.reshape(Y, newshape=(-1, 1))), axis=-1)

    np.random.seed(30101990)
    np.random.shuffle(tot)

    X = tot[:, :-1]
    Y = tot[:, -1]

    X_train = X[:num_train_samples]
    Y_train = Y[:num_train_samples]

    X_test = X[num_train_samples:]
    Y_test = Y[num_train_samples:]

    return X_train, Y_train, X_test, Y_test
