from core import *


def tif_pipeline_gpr(fname_in: str, fname_out_mean: str, fname_out_var: str, **kwargs):
    from gaussian_process_regression import rbf_regression_over_large_array
    import image_processing

    image = image_processing.import_tif_file(file_name=fname_in)

    mean, var = rbf_regression_over_large_array(image, **kwargs)

    image_processing.export_tif_file(fname_out_mean, mean)
    image_processing.export_tif_file(fname_out_var, var)


def tif_pipeline_kde(fname_in: str, fname_out: str, **kwargs):
    from kernel_density_estimation import gaussian_kernel_density_estimation
    from image_processing import import_tif_file, export_tif_file

    image = import_tif_file(file_name=fname_in)

    estimated = gaussian_kernel_density_estimation(image, **kwargs)

    estimated = np.log(estimated/2+1.)

    export_tif_file(fname_out, estimated, np.dtype("uint16"))
