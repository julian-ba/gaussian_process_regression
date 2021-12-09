import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf


def simple_rbf_regression(x, fx, variance=1., lengthscales=1.):
    rbf_model = gpf.models.GPR(
        data=(x, fx),
        kernel=gpf.kernels.stationaries.SquaredExponential(variance=variance, lengthscales=lengthscales),
    )
    rbf_model.likelihood.variance.assign(0.2 * max(abs(fx[0])))
    return rbf_model

