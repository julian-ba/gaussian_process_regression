import numpy as np
import gpflow as gpf


def simple_rbf_regression(x, fx, variance=1., lengthscales=1, noise_value=None):
    if noise_value is None:
        noise_value = max(np.abs(fx))[0] * 0.2
    rbf_model = gpf.models.GPR(
        data=(x, fx),
        kernel=gpf.kernels.stationaries.SquaredExponential(variance=variance, lengthscales=lengthscales),
    )
    rbf_model.likelihood.variance.assign(noise_value)
    return rbf_model

