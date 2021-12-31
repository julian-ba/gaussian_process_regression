from scipy import stats
import numpy as np
from core import *


def automatic_gaussian(x):
    mean = x.mean
    std = x.std

    def gaussian_with_mean_and_var(_x):
        return stats.norm.pdf(_x, loc=mean, scale=std)

    return gaussian_with_mean_and_var


def linear(a, b):

    def linear_function(x):
        return (x - a)/b

    return linear_function


def kernel_density_estimator(x, h, kernel):
    n = len(x)

    def estimator(_x):

        cumulative_sum = 0
        for i in range(n):
            cumulative_sum += kernel(linear(x[i], h)(_x))

        return cumulative_sum / (n*h)

    return estimator

