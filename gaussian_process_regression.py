import numpy as np

import image_processing


def rbf_regression_model(x, fx, variance, lengthscales, noise_value=None):
    from gpflow import models, kernels
    from core import exactly_2d
    from numpy import amax, abs
    x = exactly_2d(x=x)
    fx = exactly_2d(x=fx)

    if noise_value is None:
        noise_value = amax(abs(fx), initial=0.001) * 0.2
    rbf_model = models.GPR(
        data=(x, fx),
        kernel=kernels.stationaries.SquaredExponential(variance=variance, lengthscales=lengthscales),
    )
    rbf_model.likelihood.variance.assign(noise_value)

    return rbf_model


def rbf_regression(x, threshold, variance, lengthscales, step=None, evaluate_at=None, considered_at=None):
    from core import Grid, sparsify
    if evaluate_at is None:
        evaluate_at = x
    grid = Grid(evaluate_at, step=step)
    _x, fx = sparsify(x, threshold, slices=considered_at, step=step)
    if fx.size == 0:
        return np.zeros(grid.shape), np.zeros_like(grid.shape)
    else:
        grid_list = grid.get_list(dtype=float)
        mean, var = rbf_regression_model(_x, fx, variance, lengthscales).predict_f(grid_list)
        mean = mean.numpy()
        var = var.numpy()
        return grid.to_array(mean), grid.to_array(var)


def rbf_regression_over_large_array(x, threshold, variance, lengthscales, step=None):
    from core import LargeArrayIterator
    output = np.empty_like(x)
    for slices in LargeArrayIterator(x, 50, np.ceil(5*lengthscales)):
        output[slices.evaluate] = rbf_regression(
            x[slices.consider], threshold, variance, lengthscales, step, slices.evaluate, slices.consider)[0]

    return output

