from core import *
import numpy as np
import scipy.stats as stats


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


def generate_splits(number_of_splits, array):
    # Probably could be handled more efficiently
    internal_array = array.copy()
    q, r = np.divmod(len(internal_array), number_of_splits)
    list_of_splits = []
    for i in range(number_of_splits):
        indices_array = np.empty(q + min(max(0, r), 1))
        with np.nditer(indices_array, op_flags=["readwrite"]) as it:
            for j in it:
                index_to_pop = np.random.randint(0, len(internal_array))
                j[...] = internal_array[index_to_pop]
                internal_array = np.delete(internal_array, index_to_pop)
        list_of_splits.append(indices_array.astype(np.dtype("uint16")))
        r -= 1

    return tuple(list_of_splits)


def random_indices(number_of_splits, upper_index):
    return generate_splits(number_of_splits, np.arange(upper_index))


def norm_1(x, y):
    return np.sum(np.abs(x - y))


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
        rounded_to_zero = norm(mean, scale).cdf(np.array([[[0.5]]]))
        rounded_to_one = np.ones_like(rounded_to_zero) - rounded_to_zero
        weights = np.stack(rounded_to_zero, rounded_to_one)
        return np.array([0, 1]), weights

    elif distribution_representation == "samples":
        raise NotImplementedError

    else:
        raise ValueError("distribution_representation must be \"pdf\", \"cdf\", or \"samples\".")



def _energy_distance_wrapper_between_gpr_and_validation_1d_according_to_num(num):
    from scipy.stats import energy_distance
    def energy_distance_between_gpr_and_validation_1d(gpr_mean, gpr_var, *validation):
        points, weights = generate_gaussian_points_with_weights(gpr_mean, np.sqrt(gpr_var), num=num)
        return energy_distance(points, validation, weights)
    return energy_distance_between_gpr_and_validation_1d


def energy_distance_for_gpr(validation_distributions, gpr_prediction, num:int=5) -> np.ndarray:
    energy_distance_ufunc = np.frompyfunc(
        _energy_distance_wrapper_between_gpr_and_validation_1d_according_to_num(num),
        nin=2+len(validation_distributions),
        nout=1
    )
    return energy_distance_ufunc(gpr_prediction[0], gpr_prediction[1], *validation_distributions)


def optimized_cumulative_energy_distance_for_gpr(validation_distributions, gpr_prediction, scoring_weight=1., **kwargs) -> float:
    from scipy.stats import energy_distance
    points, weights = generate_gaussian_points_with_weights(gpr_prediction[0], gpr_prediction[1], **kwargs)
    big_validation_array = np.stack(
        [np.isclose(validation_distribution, np.zeros_like(validation_distribution))
         for validation_distribution in validation_distributions]
    )
    num = len(points)
    distance = 0.
    zero = np.all(big_validation_array, axis=0)
    zero_where = tuple([slice(None, None, None)]+list(zero.nonzero()))
    num_of_zeros = len(zero_where[1])
    zero_weights = weights[zero_where].sum(axis=1)
    distance += energy_distance(points, [0], zero_weights, None) * num_of_zeros * scoring_weight

    non_zero = np.logical_not(zero)
    non_zero_where = tuple([slice(None, None, None)] + list(non_zero.nonzero()))
    non_zero_gaussian_distributions = weights[non_zero_where]
    non_zero_validation_distribution = big_validation_array[non_zero_where]
    for idx in range(non_zero_validation_distribution.shape[1]):
        gaussian_weights = non_zero_gaussian_distributions[:, idx]
        if np.all(np.isclose(gaussian_weights, np.zeros(num))):
            distance += energy_distance(points, non_zero_validation_distribution[:, idx])
        else:
            distance += energy_distance(
                points, non_zero_validation_distribution[:, idx], gaussian_weights, None
            )

    return distance


def optimized_cumulative_energy_distance_for_kde(validation_distributions, kde_predictions, scoring_weight=1.) -> float:
    from scipy.stats import energy_distance
    big_validation_array = np.stack(
        [np.isclose(validation_distribution, np.zeros_like(validation_distribution))
         for validation_distribution in validation_distributions]
    )
    zero = np.all(big_validation_array, axis=0)
    distance = 0.
    zero_where = tuple([slice(None, None, None)] + list(zero.nonzero()))
    big_kde_predictions = np.stack(kde_predictions)
    num_of_zeros = len(zero_where[1])
    points, weights = np.unique(big_kde_predictions[zero_where], return_counts=True)
    distance += energy_distance(points, [0], weights, None) * num_of_zeros * scoring_weight



    non_zero = np.logical_not(zero)
    non_zero_where = tuple([slice(None, None, None)] + list(non_zero.nonzero()))
    non_zero_kde_distributions = big_kde_predictions[non_zero_where]
    non_zero_validation_distribution = big_validation_array[non_zero_where]
    for idx in range(non_zero_validation_distribution.shape[1]):
        points, weights = np.unique(non_zero_kde_distributions[:, idx], return_counts=True)
        distance += energy_distance(
            points, non_zero_validation_distribution[:, idx], weights, None
        )

    return distance


def euclidean_distribution_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))
    

def cross_val_run(file_names, n, output_image=False):
    from gaussian_process_regression import rbf_regression_over_large_array
    from kernel_density_estimation import gaussian_kernel_density_estimation
    from image_processing import array_crops, export_tif_file, import_tif_file
    ffish_crops = array_crops(*import_tif_file(*file_names))
    fish_shape = shape_from_slice(*ffish_crops[0][1:])
    middle_z = [(i[0].stop - i[0].start)//2 for i in ffish_crops]
    output = np.empty((n, 2))
    for i in range(n):
        splits1, splits2 = random_indices(2, len(file_names))

        agglomerated_fish = np.zeros(fish_shape)
        for j in splits1:
            agglomerated_fish += import_tif_file(file_names[j], dtype=float, key=middle_z[j])[ffish_crops[j][1:]]
        agglomerated_fish /= len(splits1)
        gpr = rbf_regression_over_large_array(agglomerated_fish, lengthscales=20, step=(1, 1))[0]
        kde = gaussian_kernel_density_estimation(agglomerated_fish, (20, 20))

        agglomerated_fish = np.zeros(fish_shape)
        for j in splits2:
            agglomerated_fish += import_tif_file(file_names[j], dtype=float, key=middle_z[j])[ffish_crops[j][1:]]
        agglomerated_fish /= len(splits2)
        if output_image:
            if i < 4:
                export_tif_file("figures/run{}_gpr".format(i), gpr)
                export_tif_file("figures/run{}_kde".format(i), kde)

        output[i, 0] = norm_1(gpr, agglomerated_fish)
        output[i, 1] = norm_1(kde, agglomerated_fish)

    return output


def finite_sum(*array):
    return np.stack(array).sum(axis=0)

def average(*array):
    return np.average(np.stack(array), axis=0)


def optimize_parameters(
        file_names, optimize_func, distance, var_kwargs,
        const_kwargs=None, agglomeration_func=average, iterations=10, output=False, normalize=False
):
    from image_processing import array_crops, import_tif_file, export_tif_file
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
    from image_processing import array_crops, import_tif_file, export_tif_file
    from gaussian_process_regression import rbf_regression_over_large_array
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
    from image_processing import array_crops, import_tif_file, export_tif_file
    from kernel_density_estimation import gaussian_kernel_density_estimation
    from tqdm import trange
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
            param_score[idx] += optimized_cumulative_energy_distance_for_kde(validation_arrays, kde_predictions, **kwargs)
    param_score /= iterations
    dtype = np.dtype([("sigma", sigma_range.dtype), ("score", np.dtype(float))])
    return np.array([(sigma_range[idx], param_score[idx]) for idx in range(len(sigma_range))], dtype=dtype)
