from core import *


def tif_pipeline_gpr(fname_in, fname_out_mean, fname_out_var):
    import gaussian_process_regression as gpr
    import image_processing

    image = image_processing.import_tif_file(fname=fname_in)

    mean = np.zeros_like(image)

    lengthscale = 10
    threshold = 0.05
    noise = np.amax(image) * 0.2
    var = np.full_like(image, np.nan)

    for i in image_processing.subdivided_array_and_considered_part_slices(image, 30, 5*lengthscale):
        # Subdivide image into parts; we create a model for each part due to memory constraints.
        shape_of_image_part = image_processing.shape_from_slice(*i[0])

        x, fx = image_processing.sparsify(image[i[1]], threshold, *i[1])  # The points with which we generate each model

        if x.size == 0:  # If there are no points, we pass (mean is 0)
            pass
        else:  # Otherwise, we generate a model using x, fx
            model = gpr.rbf_regression_model(x=x, fx=fx, lengthscales=lengthscale, noise_value=noise, variance=1)
            grid_coord_list = coord_or_index_list(*i[0]).astype(np.dtype(float))
            mean_part, var_part = model.predict_f(grid_coord_list)
            mean_part = mean_part.numpy().reshape(shape_of_image_part)
            var_part = var_part.numpy().reshape(shape_of_image_part)
            mean[i[0]] = mean_part
            var[i[0]] = var_part

    var = np.nan_to_num(var, np.amax(var))

    image_processing.export_tif_file(fname_out_mean, mean, fit=True)
    image_processing.export_tif_file(fname_out_var, var, fit=True)


def tif_pipeline_kde(fname_in, fname_out):
    from kernel_density_estimation import gaussian_kernel_density_estimation
    from image_processing import import_tif_file, export_tif_file

    image = import_tif_file(fname=fname_in)

    estimated = gaussian_kernel_density_estimation(image, (3, 9, 9))

    estimated = np.log(estimated/2+1.)

    export_tif_file(fname_out, estimated, np.dtype("uint16"), fit=True)
