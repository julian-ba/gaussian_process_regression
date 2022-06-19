import gpflow
import numpy as np
from user_values import MEMORY_STEP_2D



def rbf_regression_model(x: np.ndarray, fx: np.ndarray, lengthscales=1., variance=1., float_type=np.float32, noise_value=None, do_optimization=True) -> gpflow.models.GPModel:
    import tensorflow as tf
    from core import exactly_2d
    from numpy import amax, abs

    with gpflow.config.as_context(temporary_config=gpflow.config.Config(float=np.float32)):
        x, fx = exactly_2d(x.astype(dtype=float_type), fx.astype(dtype=float_type))

        if noise_value is None:
            noise_value = amax(abs(fx), initial=1e-20) * 0.2
        rbf_model = gpflow.models.GPR(
            data=(x, fx),
            kernel=gpflow.kernels.stationaries.SquaredExponential(variance=variance, lengthscales=lengthscales),
        )
        rbf_model.likelihood.variance.assign(noise_value)

        if do_optimization:
            gpflow.set_trainable(rbf_model.kernel.lengthscales, False)

            opti = tf.optimizers.Adam()
            opti.minimize(rbf_model.training_loss, rbf_model.trainable_variables)

    return rbf_model


def _predict_distribution(model, grid):
    mean, var = model.predict_f(grid.get_list(float))
    mean = mean.numpy()
    var = var.numpy()
    return grid.to_array(mean, var)


def rbf_regression(
        *arrays: np.ndarray, threshold=0.15, noise_value=0.2, step=None, evaluate_at=None, considered_at=None, fill="var", include_zeros=True,
        **kwargs
) -> tuple:
    from core import Grid, sparsify
    assert all([array.shape == arrays[0].shape for array in arrays])
    shape = arrays[0].shape
    if evaluate_at is None:
        grid = Grid(shape, step=step)
    else:
        grid = Grid(*evaluate_at, step=step)
    _x, fx = sparsify(*arrays, threshold=threshold, slices=considered_at, step=step, include_zeros=include_zeros)
    if fx.size == 0:
        if fill == "var":
            fill = noise_value
        return np.zeros(grid.shape), np.full(grid.shape, fill)

    else:
        return _predict_distribution(model=rbf_regression_model(_x, fx, noise_value=noise_value, **kwargs), grid=grid)


def rbf_regression_over_large_array(*array: np.ndarray, threshold=0.05, lengthscales=1., step=None, it_step=MEMORY_STEP_2D, method:str="fast", include_zeros=True, **kwargs) -> tuple:
    from core import LargeArrayIterator
    shape = array[0].shape
    assert all([array_i.shape == shape for array_i in array])
    output_mean = np.empty(shape=shape, dtype=float)
    output_var = np.empty(shape=shape, dtype=float)

    if method == "fast":  # ~3-4 times quicker, but much less accurate
        if step is not None:
            step = np.array(step)
        else:
            step = np.array(1)
        epsilon = np.ceil(10 * np.power(np.divide(lengthscales, step), 2)).astype(np.dtype(int))
        for i in LargeArrayIterator(array[0], it_step, epsilon=epsilon):
            output_mean[i.evaluate], output_var[i.evaluate] = rbf_regression(
                *[xi[i.consider] for xi in array], threshold=threshold, step=step, lengthscales=lengthscales,
                evaluate_at=i.evaluate, considered_at=i.consider, include_zeros=include_zeros
            )

    elif method == "slow":  # Perfectly accurate
        from core import sparsify, Grid
        x, fx = sparsify(*array, threshold=threshold, step=step, include_zeros=include_zeros)
        model = rbf_regression_model(x=x, fx=fx, lengthscales=lengthscales, **kwargs)

        for i in LargeArrayIterator(array[0], it_step):
            grid = Grid(*i.evaluate, step=step)
            output_mean[i.evaluate], output_var[i.evaluate] = _predict_distribution(model=model, grid=grid)

    return output_mean, output_var


def maximum_likelihood_rbf_regression(*array, threshold, step=None, evaluate_at=None, considered_at=None, **kwargs):
    from core import Grid, sparsify
    if evaluate_at is None:
        grid = Grid(array)
    else:
        grid = Grid(evaluate_at, step=step)
    _x, fx = sparsify(*array, threshold, slices=considered_at, step=step)
    if fx.size == 0:
        return np.zeros(grid.shape)

    else:
        grid_list = grid.get_list(dtype=float)
        mean = rbf_regression_model(_x, fx, **kwargs).predict_f(grid_list)[0].numpy()
        return grid.to_array(mean)

def maximum_likelihood_rbf_regression_over_large_array(
        *array, threshold, lengthscales=1., step=None, it_step=None, **kwargs
):
    from core import LargeArrayIterator
    assert all(xi.shape == array[0].shape for xi in array)
    shape = array[0].shape
    ndim = len(shape)
    output_mean = np.empty(shape=shape, dtype=float)
    if step is not None:
        step = np.array(step)
        index_lengthscales = np.array(lengthscales) / step
        if it_step is None:
            it_step = step * 5
    else:
        index_lengthscales = np.array(lengthscales).astype(int)
        if it_step is None:
            it_step = np.array([150] * ndim)
    it = LargeArrayIterator(array, it_step, np.ceil(10 * index_lengthscales))
    for i in range(len(it)):
        output_mean[it[i].evaluate] = maximum_likelihood_rbf_regression(
            array[it[i].consider], threshold, step=step, lengthscales=lengthscales,
            evaluate_at=it[i].evaluate, considered_at=it[i].consider, **kwargs
        )
    return output_mean