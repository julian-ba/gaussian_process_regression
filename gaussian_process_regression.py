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


def regress_over_large_array(array, threshold, step, lengthscale=(3, 9, 9), variance=1., **kwargs):
    from image_processing import sparsify, subdivided_array_and_considered_part_slices
    step = np.broadcast_to(step, array.ndim)
    output = np.zeros_like(array)
    for i in subdivided_array_and_considered_part_slices(array, (15, 45, 45), resize_to_indices(lengthscale, step)):
        x, fx = sparsify(array[i[1]], threshold, *xis_from_slice_and_step(step, *i[1]), **kwargs)
        if x.size == 0:
            pass
        else:
            model = rbf_regression_model(x, fx, variance, lengthscale, noise_value=0.2)
            output[i[0]] = np.reshape(
                model.predict_f(coord_or_index_list(*xis_from_slice_and_step(step, *i[0])))[0].numpy(),
                output[i[0]].shape
            )

    return output

