import numpy as np
import os
import gzip


def load_mnist_digits():
    '''
    Load MNIST (original, with digits)

    Returns
    -------------------------
    :return:
        x_train with shape (num_train_samples, h, w)
        y_train with shape (num_train_samples,)
        x_test with shape (num_test_samples, h, w)
        y_test with shape (num_test_samples,)
        label_dict
    '''

    print("Loading MNIST file...")
    x_train = np.load('mnist/x_train.npy')
    y_train = np.load('mnist/y_train.npy')
    x_test = np.load('mnist/x_test.npy')
    y_test = np.load('mnist/y_test.npy')

    # Costruisco le 10 label per le 10 cifre
    label_dict = {i: str(i) for i in range(0, 10)}

    return x_train, y_train, x_test, y_test, label_dict;


def load_mnist_fashion():
    '''
    Load fashion MNIST dataset

    Returns:
    -----------------------------------
    :return:
        x_train with shape (num_train_samples, h, w)
        y_train with shape (num_train_samples,)
        x_test with shape (num_test_samples, h, w)
        y_test with shape (num_test_samples,)
        label_dict
    '''

    path = 'mnist'
    x_train_path = os.path.join(path, 'train-images-idx3-ubyte.gz')
    y_train_path = os.path.join(path, 'train-labels-idx1-ubyte.gz')
    x_test_path = os.path.join(path, 'test-images-idx3-ubyte.gz')
    y_test_path = os.path.join(path, 'test-labels-idx1-ubyte.gz')

    with gzip.open(y_train_path, 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(x_train_path, 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(y_train), 28, 28)
    with gzip.open(y_test_path, 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(x_test_path, 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(y_test), 28, 28)

    label_dict = {0: 'T-shirt/top',
                  1: 'Trouser',
                  2: 'Pullover',
                  3: 'Dress',
                  4: 'Coat',
                  5: 'Sandal',
                  6: 'Shirt',
                  7: 'Sneaker',
                  8: 'Bag',
                  9: 'Ankle boot'
                  }

    return x_train, y_train, x_test, y_test, label_dict;


def load_mnist(which_type, threshold=0.5):
    '''

    :param which_type:
    :param threshold:
    :return:
    '''

    assert which_type in ['digits', 'fashion'], 'Not valid MNIST type: {}'.format(which_type)

    if which_type == 'digits':
        x_train, y_train, x_test, y_test, label_dict = load_mnist_digits()
    else:
        x_train, y_train, x_test, y_test, label_dict = load_mnist_fashion()

    # Trasformo i valori reali in discreti, 0 o 1
    x_train = np.float32(x_train) / 255.
    x_train[x_train >= threshold] = 1
    x_train[x_train < threshold] = 0

    x_test = np.float32(x_test) / 255.
    x_test[x_test >= threshold] = 1
    x_test[x_test < threshold] = 0

    return x_train, y_train, x_test, y_test, label_dict
