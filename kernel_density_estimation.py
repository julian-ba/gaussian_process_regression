from scipy import stats
import numpy as np
from core import *


def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def euclidean_norm(x):
    return euclidean_distance(x, x)


def to_radial_function(function):

    def radial_function(x):
        return function(euclidean_norm(x))

    return radial_function


def gaussian(mean, std):
    def gaussian_with_mean_and_var(_x):
        return stats.norm.pdf(_x, loc=mean, scale=std)
    return gaussian_with_mean_and_var


def automatic_gaussian(x):
    mean = x.mean
    std = x.std

    return gaussian(mean, std)


def bandwidth_rule_of_thumb(n, std):
    if n == 0:
        return 0.
    else:
        return np.power(4 * std**5 / (3*n), 1/5)


def linear(a, b):

    def linear_function(x):
        return (x - a)/b

    return linear_function


def kernel_density_estimator(x, h, kernel):
    n = len(x)

    def estimator(_x):

        if n == 0:
            return 0.

        else:
            cumulative_sum = 0.
            for i in range(n):
                cumulative_sum += kernel(linear(x[i], h)(_x))
            return cumulative_sum / (n*h)

    return estimator

