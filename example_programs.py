from core import *


def tif_pipeline_gpr(fname_in, fname_out_mean, fname_out_var, **kwargs):
    from gaussian_process_regression import rbf_regression_over_large_array
    import image_processing

    image = image_processing.import_tif_file(fname=fname_in)

    lengthscale = 10
    threshold = 0.05

    mean, var = rbf_regression_over_large_array(image, threshold, lengthscale, **kwargs)

    image_processing.export_tif_file(fname_out_mean, mean, fit=True)
    image_processing.export_tif_file(fname_out_var, var, fit=True)


def tif_pipeline_kde(fname_in, fname_out):
    from kernel_density_estimation import gaussian_kernel_density_estimation
    from image_processing import import_tif_file, export_tif_file

    image = import_tif_file(fname=fname_in)

    estimated = gaussian_kernel_density_estimation(image, (3, 9, 9))

    estimated = np.log(estimated/2+1.)

    export_tif_file(fname_out, estimated, np.dtype("uint16"), fit=True)
