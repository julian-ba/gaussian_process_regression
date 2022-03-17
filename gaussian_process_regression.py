import gpflow.models
import numpy as np


def rbf_regression_model(x: np.ndarray, fx: np.ndarray, lengthscales=1., variance=1., noise_value=None) -> gpflow.models.GPModel:
    from gpflow import models, kernels, optimizers, set_trainable
    from core import exactly_2d
    from numpy import amax, abs
    x = exactly_2d(x=x)
    fx = exactly_2d(x=fx)

    if noise_value is None:
        noise_value = amax(abs(fx), initial=1e-20) * 0.2
    rbf_model = models.GPR(
        data=(x, fx),
        kernel=kernels.stationaries.SquaredExponential(variance=variance, lengthscales=lengthscales),
    )
    rbf_model.likelihood.variance.assign(noise_value)

    set_trainable(rbf_model.kernel.lengthscales, False)

    opti = optimizers.Scipy()
    opti.minimize(rbf_model.training_loss, variables=rbf_model.trainable_variables)

    return rbf_model


def rbf_regression(
        x, threshold,
        variance=1., step=None, evaluate_at=None, considered_at=None, fill="var",
        **kwargs
):
    from core import Grid, sparsify
    if evaluate_at is None:
        grid = Grid(x, step=step)
    else:
        grid = Grid(evaluate_at, step=step)
    _x, fx = sparsify(x, threshold, slices=considered_at, step=step)
    if fx.size == 0:
        if fill == "var":
            fill = variance
        return np.zeros(grid.shape), np.full(grid.shape, fill)

    else:
        grid_list = grid.get_list(dtype=float)
        mean, var = rbf_regression_model(_x, fx, variance=variance, **kwargs).predict_f(grid_list)
        mean = mean.numpy()
        var = var.numpy()
        return grid.to_array(mean), grid.to_array(var)


def rbf_regression_over_large_array(x, threshold, lengthscales=1., step=None, it_step=None, **kwargs):
    from core import LargeArrayIterator
    output_mean = np.empty_like(x, dtype=float)
    output_var = np.empty_like(x, dtype=float)
    if step is not None:
        step = np.array(step)
        index_lengthscales = np.array(lengthscales) / step
        if it_step is None:
            it_step = step * 5
    else:
        index_lengthscales = np.array(lengthscales).astype(int)
        if it_step is None:
            it_step = np.array([150]*x.ndim)
    it =  LargeArrayIterator(x, it_step, np.ceil(10*index_lengthscales))
    for i in range(len(it)):
        output_mean[it[i].evaluate], output_var[it[i].evaluate] = rbf_regression(
            x[it[i].consider], threshold, step=step, lengthscales=lengthscales,
            evaluate_at=it[i].evaluate, considered_at=it[i].consider, **kwargs
        )
    return output_mean, output_var


def maximum_likelihood_rbf_regression(x, threshold, step=None, evaluate_at=None, considered_at=None, **kwargs):
    from core import Grid, sparsify
    if evaluate_at is None:
        grid = Grid(x)
    else:
        grid = Grid(evaluate_at, step=step)
    _x, fx = sparsify(x, threshold, slices=considered_at, step=step)
    if fx.size == 0:
        return np.zeros(grid.shape)

    else:
        grid_list = grid.get_list(dtype=float)
        mean = rbf_regression_model(_x, fx, **kwargs).predict_f(grid_list)[0].numpy()
        return grid.to_array(mean)

def maximum_likelihood_rbf_regression_over_large_array(
        x, threshold, lengthscales=1., step=None, it_step=None, **kwargs
):
    from core import LargeArrayIterator
    output_mean = np.empty_like(x, dtype=float)
    if step is not None:
        step = np.array(step)
        index_lengthscales = np.array(lengthscales) / step
        if it_step is None:
            it_step = step * 5
    else:
        index_lengthscales = np.array(lengthscales).astype(int)
        if it_step is None:
            it_step = np.array([150] * x.ndim)
    it = LargeArrayIterator(x, it_step, np.ceil(10 * index_lengthscales))
    for i in range(len(it)):
        output_mean[it[i].evaluate] = maximum_likelihood_rbf_regression(
            x[it[i].consider], threshold, step=step, lengthscales=lengthscales,
            evaluate_at=it[i].evaluate, considered_at=it[i].consider, **kwargs
        )
    return output_mean