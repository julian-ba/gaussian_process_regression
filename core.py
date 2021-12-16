# TODO: Create function which returns a list of the coordinates of a grid.
import numpy as np
import gpflow as gpf
import numbers


def rbf_regression(x, fx, variance=1., lengthscales=1, noise_value=None):
    x = atleast_column(x=x)
    fx = atleast_column(x=fx)

    if noise_value is None:
        noise_value = max(np.abs(fx))[0] * 0.2
    rbf_model = gpf.models.GPR(
        data=(x, fx),
        kernel=gpf.kernels.stationaries.SquaredExponential(variance=variance, lengthscales=lengthscales),
    )
    rbf_model.likelihood.variance.assign(noise_value)

    return rbf_model


def atleast_column(x):

    try:
        x = np.asarray(x)
        if x.ndim == 1:
            output_array = x[:, np.newaxis]
        else:
            output_array = x
    except TypeError:
        raise TypeError("The input must be a NumPy-array.")

    return output_array


# def flatten_values(x, output_dim=1):


def coord_list(*xi):
    coord_grid = np.reshape(np.stack(np.meshgrid(*xi), axis=-1), (-1, len(xi)))
    return coord_grid


def coord_list_and_meshgrid(*xi):
    meshxi = np.meshgrid(*xi)
    coord_grid = np.reshape(np.stack(meshxi, axis=-1), (-1, len(xi)))
    return coord_grid, meshxi
