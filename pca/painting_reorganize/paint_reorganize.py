import numpy as np
import os
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image

plt.ion()


def read_image(path):
    filename = os.path.expanduser(path)
    image = scipy.misc.imread(filename, mode="RGBA")[:, :, :3]
    return image


def save_image(array, path):
    filename = os.path.expanduser(path)
    im = Image.fromarray(array)
    im.save(filename)


def pca_algorithm(X, percentual):
    mean = np.mean(X, axis=0)
    X = X - mean

    eigVal, eig_vet = np.linalg.eig(np.cov(X.T))
    eig_vet = eig_vet.T

    def num_eig_vals(eigVals, percentual):
        eigVals = np.sort(eigVals)[-1::-1]
        sum_eigVals = np.sum(eigVals)
        tmp = num = 0
        for i in eigVals:
            tmp += i
            num += 1
            if tmp >= sum_eigVals * percentual:
                return num

    num_component = num_eig_vals(eigVal, percentual)
    eig_vet = eig_vet[0:num_component]

    X = np.dot(X, eig_vet.T)
    return np.negative(X[:, 0])


def get_new_image(old_image):
    pixels = old_image.reshape((old_image.shape[0] * old_image.shape[1], 3))
    pixels_pca_component = np.squeeze(pca_algorithm(pixels, percentual= 0.99))

    pixels_by_first_component = pixels[np.argsort(pixels_pca_component)]

    num_rows, num_columns = old_image.shape[0], old_image.shape[1]
    new_image = np.zeros_like(old_image)

    old_column = None
    for column in range(num_columns):
        pixel_in_column = pixels_by_first_component[num_rows * column:(column + 1) * num_rows]
        first_column_component = np.squeeze(pca_algorithm(pixel_in_column, percentual=0.99))
        new_column = pixel_in_column[np.argsort(first_column_component)]

        # Now figure out which direction will mach the previous column better
        if old_column is not None:
            distance = np.mean(np.sqrt(np.sum((new_column - old_column) ** 2, 1)))
            flipped_distance = np.mean(np.sqrt(np.sum((new_column[-1::-1] - old_column) ** 2, 1)))
            if flipped_distance < distance:
                new_column = new_column[-1::-1]
        old_column = new_column

        new_image[:, column, :] = new_column
    return new_image


if __name__ == '__main__':
    num = 23

    old_image = []
    new_image = []
    for i in range(num):
        old_image.append(read_image('./input/input_{}.jpg'.format(i)))
        new_image.append(get_new_image(old_image[i]))
        save_image(new_image[i], path='./output/output_{}.jpg'.format(i))
        print("Trasformazione {} su {} realizzata.".format(i + 1, num))


    _, (ax0, ax1) = plt.subplots(1, 2)
    for i in range(num):
        ax0.imshow(old_image[i])
        ax0.set_title("Dipinto")
        ax1.imshow(new_image[i])
        ax1.set_title("Palette")
        plt.waitforbuttonpress()
