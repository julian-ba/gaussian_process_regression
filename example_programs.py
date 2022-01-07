import gpflow.utilities

from core import *
import image_processing
from skimage import io
import tifffile
import gaussian_process_regression as gpf
import gpflow
import numpy as np


def tif_pipeline_gpr(fname_in, fname_out):
    # from core import *
    # from skimage import io
    # import gaussian_process_functions as gpf
    # import image_processing

    image = image_processing.import_tif_file(fname=fname_in)
    print(image.shape)

    mean = np.empty_like(image)

    lengthscale = 10
    threshold = 0.05

    for i in subdivided_array_slices(image, step_size=50):  # Subdivide image into parts; we create a model for each
        # part due to memory constraints.
        shape_of_image_part = tuple([j.stop - j.start for j in i])
        slices_of_considered_part = []
        for j in range(len(i)):  # For creating each model, we only consider points above threshold which are
            # within 2 * lengthscale of the bounds of the part of the image we are generating. We assume the
            # contribution of points outside this area to the calculated mean is negligible.
            lower = i[j].start - int(np.ceil(2*lengthscale))
            if lower < 0:
                lower = 0
            upper = i[j].stop + int(np.ceil(2*lengthscale))
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
            model = gpf.rbf_regression(x=x, fx=fx, lengthscales=lengthscale)
            grid_coord_list = coord_or_index_list(*i).astype(np.dtype(float))
            mean_part = model.predict_f(grid_coord_list)[0].numpy().reshape(shape_of_image_part)
            mean[i] = mean_part

    mean *= 100  # Scale mean to show more contrast.
    print(mean.shape)
    x = subdivided_array_slices(mean, (1, 5, 5))
    for i in range(1000):
        print(mean[x[i]])
    tifffile.imsave(fname_out, mean)  # Save image

