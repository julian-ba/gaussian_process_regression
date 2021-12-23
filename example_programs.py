from core import *
import image_processing
from skimage import io
import gaussian_process_functions as gpf


def tif_pipeline_gpr(fname_in, fname_out):
    # from core import *
    # from skimage import io
    # import gaussian_process_functions as gpf
    # import image_processing
    image = image_processing.import_tif_file(fname=fname_in)

    x, fx = sparsify(image, 0.01)

    model = gpf.rbf_regression(x=x, fx=fx, lengthscales=5)

    mean = model.predict_f(coord_list_evenly_spaced(*image.shape))[0].numpy().reshape(image.shape)

    io.imsave(fname_out, mean)

