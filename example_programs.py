import gpflow.utilities

from core import *
import image_processing
from skimage import io
import gaussian_process_regression as gpf
import gpflow
import numpy as np


def tif_pipeline_gpr(fname_in, fname_out):
    # from core import *
    # from skimage import io
    # import gaussian_process_functions as gpf
    # import image_processing
    image = image_processing.import_tif_file(fname=fname_in)

    mean = np.empty_like(image)

    lengthscale = 5
    threshold = 0.05
    for i in subdivided_array_slices(image, step_size=50):
        shape_of_image_part = tuple([j.stop - j.start for j in i])
        slices_of_considered_part = []
        for j in range(len(i)):
            lower = i[j].start - int(np.ceil(2*lengthscale))
            if lower < 0:
                lower = 0
            upper = i[j].stop + int(np.ceil(2*lengthscale))
            if upper > image.shape[j]:
                upper = image.shape[j]

            slices_of_considered_part.append(slice(lower, upper))

        considered_part = image[tuple(slices_of_considered_part)]
        x, fx = sparsify(considered_part, threshold, *slices_of_considered_part)
        if x.size == 0:
            mean[i] = np.zeros_like(mean[i])
        else:
            model = gpf.rbf_regression(x=x, fx=fx, lengthscales=lengthscale)
            grid_coord_list = coord_or_index_list(*i).astype(np.dtype(float))
            mean_part = model.predict_f(grid_coord_list)[0].numpy().reshape(shape_of_image_part)
            mean[i] = mean_part

    mean *= 100
    io.imsave(fname_out, mean)
