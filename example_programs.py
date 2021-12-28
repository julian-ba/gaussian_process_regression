from core import *
import image_processing
from skimage import io
import gaussian_process_functions as gpf
import numpy as np


def tif_pipeline_gpr(fname_in, fname_out):
    # from core import *
    # from skimage import io
    # import gaussian_process_functions as gpf
    # import image_processing
    image = image_processing.import_tif_file(fname=fname_in)[0]

    x, fx = sparsify(image, 0.01)

    model = gpf.rbf_regression(x=x, fx=fx, lengthscales=5)

    mean = np.empty_like(image)

    for i in subdivided_array_slices(image):
        shape = tuple([j.stop - j.start for j in i])
        mean[i] = model.predict_f(slices_to_coord_list(*i).astype(np.dtype(float)))[0].numpy().reshape(shape)

    mean *= 100

    io.imsave(fname_out, mean)
