import scipy.stats as stats

from core import *


def smooth_data(n, loc=0., scale=1.):
    # Returns n sample points from a multivariate Gaussian distribution distributed according to loc and scale.
    # The distributions in each dimension are independent.
    loc, scale = np.broadcast_arrays(loc, scale)

    array = np.empty((n, loc.size))
    for i in range(loc.size):
        array[:, i] = stats.norm.rvs(size=n, loc=loc[i], scale=scale[i])

    return array


def jagged_data(n, loc=0., scale=1) -> np.ndarray:
    # Generates n points uniformly distributed over a hyper-cuboid. loc is the lower bound of the distribution (resp.
    # dimension), loc+scale is the upper bound.
    loc, scale = np.broadcast_arrays(loc, scale)

    array = np.empty((n, loc.size))
    for i in range(loc.size):
        array[:, i] = stats.uniform.rvs(size=n, loc=loc[i], scale=scale[i])

    return array


def norm_1(x, y):
    return np.sum(np.abs(x - y))


def generate_gaussian_samples(mean, scale, num:int=1):
    if num == 1:
        return stats.norm.rvs(mean, scale)
    else:
        output = []
        for i in range(num):
            output.append(stats.norm.rvs(mean, scale))
        return tuple(output)


def generate_gaussian_points_with_weights(mean, scale, distribution_representation="pdf", num=5):
    from scipy.stats import norm
    if distribution_representation == "pdf":
        n = num // 10 + 1
        max_scale = np.amax(scale)*n*2
        points = np.linspace(0., 2*max_scale, num=num) - max_scale + np.mean(mean)
        mean_full_size = np.broadcast_to(mean, [num]+list(mean.shape))
        scale_full_size = np.broadcast_to(scale, [num]+list(mean.shape))
        return points, norm(loc=mean_full_size, scale=scale_full_size).pdf(points[:, np.newaxis, np.newaxis])

    elif distribution_representation == "cdf":
        rounded_to_zero = norm(mean, scale).cdf(np.array([[0.5]]))
        rounded_to_one = np.ones_like(rounded_to_zero) - rounded_to_zero
        weights = np.stack((rounded_to_zero, rounded_to_one))
        return np.array([0, 1]), weights

    else:
        raise ValueError("distribution_representation must be \"pdf\", or \"cdf\".")


def full_energy_distance(predictions, validations, points=None):
    from scipy.stats import energy_distance
    from collections.abc import Sequence

    if not isinstance(validations, Sequence):
        validations = tuple(validations)
    if not isinstance(predictions, Sequence):
        predictions = tuple(predictions)

    def energy_distance_1d(_prediction_and_validation):
        if points is not None:
            return energy_distance(points, _prediction_and_validation[len(predictions):], _prediction_and_validation[:len(predictions)], None)
        else:
            return energy_distance(_prediction_and_validation[:len(predictions)], _prediction_and_validation[len(predictions):])

    distance = 0.
    predictions_and_validations = np.stack(list(predictions)+list(validations))
    for idx in Grid(predictions_and_validations.shape[1:]).list:
        idx = tuple([slice(None, None, None)] + list(idx))
        distance += energy_distance_1d(predictions_and_validations[idx])
    return distance


def optimized_cumulative_energy_distance(prediction_arrays, validation_arrays, points:None|np.ndarray=None, scoring_weight=1.) -> float:
    from scipy.stats import energy_distance
    from collections.abc import Sequence

    if not isinstance(validation_arrays, Sequence):
        validation_arrays = (validation_arrays,)
    if not isinstance(prediction_arrays, Sequence):
        prediction_arrays = (prediction_arrays,)

    big_validation_array = np.stack(validation_arrays)
    zero = np.all(np.isclose(big_validation_array, 0.), axis=0)
    zero_where = tuple(
        [slice(None, None, None)] + list(zero.nonzero())
    )
    num_of_zeros = len(zero_where[1])
    distance = 0.
    big_prediction_array = np.stack(prediction_arrays)
    if num_of_zeros == 0:
        pass
    else:
        if points is not None:
            distance += energy_distance(points, [0], big_prediction_array[zero_where].sum(axis=1), None)
        else:
            distrib_points, counting_weights = np.unique(big_prediction_array[zero_where], return_counts=True)
            distance += energy_distance(distrib_points, [0], counting_weights, None) * num_of_zeros * scoring_weight

    non_zero = np.logical_not(zero)
    non_zero_where = tuple(
        [slice(None, None, None)] + list(non_zero.nonzero())
    )
    num_of_non_zeros = len(non_zero_where[1])
    if num_of_non_zeros == 0:
        pass
    else:
        non_zero_distributions = big_prediction_array[non_zero_where]
        non_zero_validation_distribution = big_validation_array[non_zero_where]
        for idx in range(non_zero_validation_distribution.shape[1]):
            if points is not None:
                distance += energy_distance(points, non_zero_validation_distribution[:, idx], non_zero_distributions[: idx], None)
            else:
                distrib_points, counting_weights = np.unique(non_zero_distributions[:, idx], return_counts=True)
                distance += energy_distance(
                    distrib_points, non_zero_validation_distribution[:, idx], counting_weights, None
                )

    return distance


def euclidean_distribution_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def average(*array):
    return np.average(np.stack(array), axis=0)


def optimize_parameters(
        file_names, optimize_func, distance, var_kwargs,
        const_kwargs=None, agglomeration_func=average, iterations=10, output=False, normalize=False
):
    from file_processing import import_tif_file, export_tif_file
    from core import array_crops
    from tqdm import tqdm
    if const_kwargs is None:
        const_kwargs = {}
    ordered_keys = list(var_kwargs.keys())
    values = list(var_kwargs.values())
    grid = Grid(values)
    param_score = np.zeros(len(grid))
    ffish_crops = array_crops(*import_tif_file(*file_names))
    middle_z = [(i[0].stop - i[0].start) // 2 for i in ffish_crops]
    num = 0
    lst = grid.get_list()
    for i in tqdm(lst):
        kwargs_i = {ordered_keys[j]: i[j] for j in range(grid.ndim)}
        kwargs_i.update(const_kwargs)
        for j in range(iterations):
            splits1, splits2 = random_indices(2, len(file_names))
            prediction = optimize_func(
            *[import_tif_file(file_names[k], key=middle_z[k])[ffish_crops[k][1:]] for k in splits1],
            **kwargs_i
            )
            if output and (j==0):
                export_tif_file(f"figures/optimization_{optimize_func.__name__}({num})", prediction)

            agglomerated_image = agglomeration_func(
                *[import_tif_file(file_names[k], key=middle_z[k])[ffish_crops[k][1:]] for k in splits2]
            )

            if normalize:
                prediction *= agglomerated_image.mean() / prediction.mean()

            param_score[num] += distance(prediction, agglomerated_image)
        num += 1
    param_score /= iterations
    dtype = [(key, var_kwargs[key].dtype) for key in ordered_keys]
    dtype.append(("score", param_score.dtype))
    dtype = np.dtype(dtype)
    lst = np.hstack((lst, exactly_2d(param_score)))
    return np.array([tuple(i) for i in lst[:]], dtype=dtype)


def optimize_gpr(file_names, lengthscales_range:np.ndarray, iterations=10, output=True, **kwargs):
    # A modified optimization run, specifically for optimizing the lengthscales parameter in GPR
    from file_processing import import_tif_file, export_tif_file
    from gaussian_process_regression import rbf_regression_over_large_array
    from core import array_crops
    from tqdm import trange
    param_score = np.zeros(len(lengthscales_range))
    image_crops = array_crops(*import_tif_file(*file_names))
    middle_z = [(i[0].stop - i[0].start) // 2 for i in image_crops]
    for idx in trange(len(lengthscales_range)):
        for num in range(iterations):
            splits1, splits2 = random_indices(2, len(file_names))
            testing_arrays = tuple(import_tif_file(file_names[k], key=middle_z[k])[image_crops[k][1:]] for k in splits1)
            prediction_gpr = rbf_regression_over_large_array(*testing_arrays, lengthscales=lengthscales_range[idx], method="slow")
            if output and (num == 0):
                export_tif_file(f"figures/output_gpr({idx})", prediction_gpr[0])
            validation_arrays = tuple(
                import_tif_file(file_names[k], key=middle_z[k])[image_crops[k][1:]] for k in splits2
            )
            param_score[idx] += optimized_cumulative_energy_distance_for_gpr(validation_arrays, prediction_gpr, **kwargs)
    param_score /= iterations
    dtype = [("lengthscales", lengthscales_range.dtype), ("score", np.dtype(float))]
    dtype = np.dtype(dtype)
    return np.array([(lengthscales_range[idx], param_score[idx]) for idx in range(len(lengthscales_range))], dtype=dtype)


def optimize_kde(file_names, sigma_range:np.ndarray, iterations=10, output=False, **kwargs):
    # A modified optimization run, specifically for optimizing the sigma parameter in KDE
    from file_processing import import_tif_file, export_tif_file
    from kernel_density_estimation import gaussian_kernel_density_estimation
    from tqdm import trange
    from core import array_crops
    param_score = np.zeros(len(sigma_range))
    ffish_crops = array_crops(*import_tif_file(*file_names))
    middle_z = [(i[0].stop - i[0].start) // 2 for i in ffish_crops]
    for idx in trange(len(sigma_range)):
        for num in range(iterations):
            splits1, splits2 = random_indices(2, len(file_names))
            testing_arrays = tuple(import_tif_file(file_names[k], key=middle_z[k], dtype=np.dtype(float))[ffish_crops[k][1:]] for k in splits1)
            kde_predictions = [
                gaussian_kernel_density_estimation(testing_array, sigma_range[idx]) for testing_array in testing_arrays
            ]
            if output and (num == 0):
                export_tif_file(f"figures/output_kde({idx})", kde_predictions[0])
            validation_arrays = tuple(
                import_tif_file(file_names[k], key=middle_z[k])[ffish_crops[k][1:]] for k in splits2
            )
            param_score[idx] += optimized_cumulative_energy_distance(validation_arrays, kde_predictions, **kwargs)
    param_score /= iterations
    dtype = np.dtype([("sigma", sigma_range.dtype), ("score", np.dtype(float))])
    return np.array([(sigma_range[idx], param_score[idx]) for idx in range(len(sigma_range))], dtype=dtype)
