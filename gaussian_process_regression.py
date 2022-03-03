import numpy as np


def rbf_regression_model(x, fx, lengthscales=1., variance=1., noise_value=None):
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


def rbf_regression(
        x, threshold,
        variance=1., step=None, evaluate_at=None, considered_at=None, fill="var",
        **kwargs
):
    from core import Grid, sparsify
    if evaluate_at is None:
        evaluate_at = x
    grid = Grid(evaluate_at, step=step)
    _x, fx = sparsify(x, threshold, slices=considered_at, step=step)
    if fx.size == 0:
        if fill == "var":
            fill = variance
        return np.zeros(grid.shape), np.full(grid.shape, fill)

    else:
        grid_list = grid.get_list(dtype=float)
        mean, var = rbf_regression_model(_x, fx, variance, **kwargs).predict_f(grid_list)
        mean = mean.numpy()
        var = var.numpy()
        return grid.to_array(mean), grid.to_array(var)


def rbf_regression_over_large_array(x, threshold, lengthscales=1., step=None, **kwargs):
    from core import LargeArrayIterator
    output_mean = np.empty_like(x, dtype=float)
    output_var = np.empty_like(x, dtype=float)
    if step is not None:
        index_lengthscales = np.array(lengthscales) / step
    else:
        index_lengthscales = lengthscales
    for slices in LargeArrayIterator(x, (1, 150, 150), np.ceil(5*index_lengthscales)):
        output_mean[slices.evaluate], output_var[slices.evaluate] = rbf_regression(
            x[slices.consider], threshold, lengthscales,
            evaluate_at=slices.evaluate, considered_at=slices.consider, **kwargs
        )

    return output_mean, output_var

