from core import *


def rbf_regression_model(x, fx, variance, lengthscales, noise_value=None):
    import gpflow as gpf
    x = exactly_2d(x=x)
    fx = exactly_2d(x=fx)

    if noise_value is None:
        noise_value = np.amax(np.abs(fx), initial=0.001) * 0.2
    rbf_model = gpf.models.GPR(
        data=(x, fx),
        kernel=gpf.kernels.stationaries.SquaredExponential(variance=variance, lengthscales=lengthscales),
    )
    rbf_model.likelihood.variance.assign(noise_value)

    return rbf_model


def rbf_regression(shape, step, x, fx, **kwargs):
    model = rbf_regression_model(x, fx, **kwargs)
    return model.predict_f(coord_array_from_shape_and_step(shape, step))[0].numpy().reshape(shape)


def rbf_regressor(**kwargs):
    return lambda evaluate_at: rbf_regression(evaluate_at, **kwargs)
