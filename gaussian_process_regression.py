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


def rbf_regression(x, threshold, variance, lengthscales, step=None, evaluate_at=None):
    from core import Grid, sparsify
    if evaluate_at is None:
        evaluate_at = x
    grid = Grid(evaluate_at, step=step)
    _x, fx = sparsify(x, threshold)
    grid_list = grid.get_list(dtype=float)
    mean, var = rbf_regression_model(_x, fx, variance, lengthscales).predict_f(grid_list)
    mean = mean.numpy()
    var = var.numpy()
    return grid.to_array(mean), grid.to_array(var)


def rbf_regression_over_large_array(x, threshold, variance, lengthscales, step=None):
    from core import subdivided_array_and_considered_part_slices
    output = np.empty_like(x)
    for slices in subdivided_array_and_considered_part_slices(x, 50, np.ceil(5*lengthscales)):
        output[slices[0]] = rbf_regression(x[slices[1]], threshold, variance, lengthscales, step, slices[0])[0]

    return output

