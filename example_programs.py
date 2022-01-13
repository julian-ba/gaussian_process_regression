from core import *


def tif_pipeline_gpr(fname_in, fname_out_mean, fname_out_var):
    import gaussian_process_regression as gpr
    import image_processing

    image = image_processing.import_tif_file(fname=fname_in)

    mean = np.empty_like(image)

    lengthscale = 10
    threshold = 0.05
    noise = np.amax(image) * 0.2
    var = np.zeros_like(image)

    for i in image_processing.subdivided_array_and_considered_part_slices(image, 30, 5*lengthscale):
        # Subdivide image into parts; we create a model for each part due to memory constraints.
        shape_of_image_part = image_processing.shape_from_slice(*i[0])

        x, fx = sparsify(image[i[1]], threshold, *i[1])  # The points with which we generate each model

        if x.size == 0:  # If there are no points, we simply set mean[i] to be 0 everywhere
            mean[i] = np.zeros_like(mean[i[0]])
        else:  # Otherwise, we generate a model using x, fx
            model = gpr.rbf_regression(x=x, fx=fx, lengthscales=lengthscale, noise_value=noise)
            grid_coord_list = coord_or_index_list(*i[0]).astype(np.dtype(float))
            mean_part, var_part = model.predict_f(grid_coord_list)
            mean_part = mean_part.numpy().reshape(shape_of_image_part)
            var_part = var_part.numpy().reshape(shape_of_image_part)
            mean[i[0]] = mean_part
            var[i[0]] = var_part

    image_processing.export_tif_file(fname_out_mean, mean, fit=True)
    image_processing.export_tif_file(fname_out_var, var, fit=True)


def tif_pipeline_kde(fname_in, fname_out):
    import kernel_density_estimation as kde
    import image_processing

    image = image_processing.import_tif_file(fname=fname_in)

    estimated = np.empty_like(image)

    threshold = 0.05
    lengthscale = 4
    epsilon = int(np.ceil(kde.find_bounds_for_gaussian(lengthscale, 1./255)))
    j = 1
    it = image_processing.subdivided_array_and_considered_part_slices(estimated, 40, epsilon)
    for i in it:
        print(str(j)+"/"+str(len(it)))
        j += 1
        shape = image_processing.shape_from_slice(*i[0])
        x, fx = sparsify(image[i[1]], threshold, *i[1])
        bandwidth = kde.bandwidth_rule_of_thumb(len(fx), lengthscale)
        if fx.size == 0:
            mean = 0.
        else:
            mean = fx.mean()
        grid_coords = coord_or_index_list(*i[0])
        image[i[0]] = kde.kernel_density_estimator(
            x, bandwidth, kde.radial_gaussian(mean, lengthscale)
        )(grid_coords).reshape(shape)

    image_processing.export_tif_file(fname_out, estimated, fit=True)
