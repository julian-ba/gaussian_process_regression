from core import *
import image_processing
from skimage import io
import gaussian_process_regression as gpr
import numpy as np
import kernel_density_estimation as kde


def tif_pipeline_gpr(fname_in, fname_out_mean, fname_out_var):
    # from core import *
    # from skimage import io
    # import gaussian_process_functions as gpf
    # import image_processing

    image = image_processing.import_tif_file(fname=fname_in)

    mean = np.empty_like(image)

    lengthscale = 10
    threshold = 0.05
    noise = np.amax(image) * 0.2
    var = np.full_like(image, noise)

    for i in subdivided_array_slices(image, step_size=30):  # Subdivide image into parts; we create a model for each
        # part due to memory constraints.
        shape_of_image_part = tuple([j.stop - j.start for j in i])
        slices_of_considered_part = []
        for j in range(len(i)):  # For creating each model, we only consider points above threshold which are
            # within 5 * lengthscale of the bounds of the part of the image we are generating. We assume the
            # contribution of points outside this area to the calculated mean is negligible.
            lower = i[j].start - int(np.ceil(5*lengthscale))
            if lower < 0:
                lower = 0
            upper = i[j].stop + int(np.ceil(5*lengthscale))
            if upper > image.shape[j]:
                upper = image.shape[j]

            slices_of_considered_part.append(slice(lower, upper))

        slices_of_considered_part = tuple(slices_of_considered_part)

        considered_part = image[slices_of_considered_part]
        x, fx = sparsify(considered_part, threshold, *slices_of_considered_part)  # The points with which we generate
        # each model
        if x.size == 0:  # If there are no points, we simply set mean[i] to be 0 everywhere
            mean[i] = np.zeros_like(mean[i])
        else:  # Otherwise, we generate a model using x, fx
            model = gpr.rbf_regression(x=x, fx=fx, lengthscales=lengthscale, noise_value=noise)
            grid_coord_list = coord_or_index_list(*i).astype(np.dtype(float))
            mean_part, var_part = model.predict_f(grid_coord_list)
            mean_part = mean_part.numpy().reshape(shape_of_image_part)
            var_part = var_part.numpy().reshape(shape_of_image_part)
            mean[i] = mean_part
            var[i] = var_part

    mean *= 255 / np.amax(mean)  # Scale mean to show more contrast.
    var *= 255 / np.amax(var)  # Scale var to show more contrast.
    io.imsave(fname_out_mean, mean.astype("uint8"))  # Save mean
    io.imsave(fname_out_var, var.astype("uint8"))  # Save var


def tif_pipeline_kde(fname_in, fname_out):
    # from core import *
    # from skimage import io
    # import kernel_density_estimation as kde
    # import image_processing

    image = image_processing.import_tif_file(fname=fname_in)

    estimated = np.empty_like(image)

    threshold = 0.05
    lengthscale = 10
    x, fx = sparsify(image, threshold)
    bandwidth = kde.bandwidth_rule_of_thumb(len(fx), lengthscale)

    estimator = np.frompyfunc(
        kde.kernel_density_estimator(x, bandwidth, kde.to_radial_function(kde.gaussian(1, lengthscale))), 1, 1
    )

    estimated = estimator(estimated)

    estimated *= 255 / np.amax(estimated)
    io.imsave(fname_out, estimated.astype("uint8"))
