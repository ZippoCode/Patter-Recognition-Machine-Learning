import numpy as np

from sklearn.datasets import make_moons


def two_moon_dataset(num_samples=100, shuffle=True, noise=None, random_state=None):
    return make_moons(num_samples, shuffle, noise, random_state)
