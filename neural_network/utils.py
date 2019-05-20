import numpy as np
import cv2
import os.path as path
import pickle

from tensorflow.examples.tutorials.mnist import input_data

EPS = np.finfo('float32').eps


def get_mnist_data(download_data_path, one_hot=True, verbose=False):
    """

    Parameters
    :param download_data_path: string
        Directory where MNIST data are downloaded and extracted
    :param one_hot: bool
        If True, targets are returned into one-hot format
    :param verbose: bool
        If True, print dataset tensors dimensions

    Returns:
    :return: Dataset
        Structure containing train, val and test MNIST dataset in a friendly format
    """

    # Download and read in MNIST dataset
    mnist = input_data.read_data_sets(download_data_path, one_hot=one_hot)
    if verbose:
        # Print image tensors shapes
        print('TRAIN tensor shape: {}'.format(mnist.train.images.shape))
        print('VAL tensor shape: {}'.format(mnist.validation.images.shape))
        print('TEST tensor shape: {}'.format(mnist.test.images.shape))

        # Print labels shape (encoded ad one-hot vectors
        print('TRAIN labels shape: {}'.format(mnist.train.labels.shape))
        print('VAL labels shape: {}'.format(mnist.validation.labels.shape))
        print('TEST labels shape: {}'.format(mnist.test.labels.shape))

    return mnist

class TilesDataset:

    def __init__(self):
        pass

    def initialize_dataset(self, dataset_root):

        # Store dataset root
        if not path.exists(dataset_root):
            raise IOError('Directory {} does not exist.'.format(dataset_root))
        self.dataset_root = dataset_root

        # Store locations of train, val and test directories
        self.train_x_dir = path.join(dataset_root, 'X_train')
        self.train_y_dir = path.join(dataset_root, 'Y_train')
        self.validation_x_dir = path.join(dataset_root, 'X_validation')
        self.validation_y_dir = path.join(dataset_root, 'Y_validation')
        self.test_x_dir = path.join(dataset_root, 'X_test')
        self.test_y_dir = path.join(dataset_root, 'T_test')

        # Number of dataset examples
        self.train_num_examples = 10000
        self.validation_num_examples = 1000
        self.test_num_examples = 1000

        # Initialize empty structure to contain data
        self.train_x = []
        self.train_y = []
        self.validation_x = []
        self.validation_y = []
        self.test_x = []
        self.test_y = []

        self._fill_data_arrays()

    def _fill_data_arrays(self):

        # Load training images
        for i in range(1, self.train_num_examples + 1):
            print('Loading training examples. {} / {}...'.format(i, self.train_num_examples))
            x_image = cv2.imread(path.join(self.train_x_dir, '{:05d}.png'.format(i)))
            y_image = cv2.imread(path.join(self.train_y_dir, '{:05d}.png'.format(i)), cv2.IMREAD_GRAYSCALE)
            self.train_x.append(x_image.astype(np.float32))
            self.train_y.append(np.expand_dims(y_image.astype(np.float32), 2))

        # Load validation examples
        for i in range(1, self.train_num_examples + 1):
            print('Loading validation examples. {} / {}...'.format(i, self.validation_num_examples))
            x_image = cv2.imread(path.join(self.validation_x_dir, '{:05d}.png'.format(i)))
            y_image = cv2.imread(path.join(self.validation_y_dir, '{:05d}.png'.format(i)), cv2.IMREAD_GRAYSCALE)
            self.validation_x.append(x_image.astype(np.float32))
            self.validation_yappend(np.expand_dims(y_image.astype(np.float32), 2))


        # Load test examples
        for i in range(1, self.train_num_examples + 1):
            print('Loading test examples. {} / {}...'.format(i, self.test_num_examples))
            x_image = cv2.imread(path.join(self.test_x_dir, '{:05d}.png'.format(i)))
            y_image = cv2.imread(path.join(self.test_y_dir, '{:05d}.png'.format(i)), cv2.IMREAD_GRAYSCALE)
            self.test_x.append(x_image.astype(np.float32))
            self.test_y.append(np.expand_dims(y_image.astype(np.float32), 2))

    def dump_to_file(self, file_path, protocol = pickle.HIGHEST_PROTOCOL):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file, protocol = protocol)

def load_tiles_dataset_from_cache(file_path):
    """

    Parameters:
    :param file_path: basestring
        Path of pickle file containing the tiles dataset dumped

    Returns:
    :return dataset: TilesDataset
        Segmentation practice data embedded into TilesDataset class
    """

    if not path.exists(file_path):
        raise IOError('File {} does not exist.'.format(file_path))
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    return dataset