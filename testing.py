from core import *
from gaussian_process_regression import rbf_regression_model
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



def time_model_1d(
        repetitions, x, fx=smooth_data, return_seconds=False, regressor=rbf_regression_model, **kwargs
):
    import time
    # Create repetitions-# of model_type of data x and return the time in ns or s taken for each resp. repetition.

    testing_points = np.linspace(0, 10, 11)

    output_array = np.empty(repetitions)
    if callable(fx):
        n = x.shape[0]
        for repetition_idx in range(repetitions):
            data = exactly_2d(fx(n=n, **kwargs))
            t1 = time.time_ns()
            regressor(x, data, **kwargs)
            t2 = time.time_ns()
            output_array[repetition_idx] = t2 - t1
    else:
        for repetition_idx in range(repetitions):
            t1 = time.thread_time_ns()
            m = regressor(x=x, fx=fx, **kwargs)
            m.predict_f_samples(testing_points)
            t2 = time.thread_time_ns()
            output_array[repetition_idx] = t2 - t1

    if return_seconds:
        return output_array * 1e-9
    else:
        return output_array



def run_test_1d(repetitions, lower_n, upper_n, step=1, *args, **kwargs):
    tested_n = np.arange(start=lower_n, stop=upper_n+1, step=step, dtype=int)
    data_array = np.empty((len(tested_n), repetitions))
    idx = 0
    for i in tested_n:
        x = exactly_2d(np.linspace(0, 10, i))
        data_array[idx] = time_model_1d(repetitions=repetitions, x=x, *args, **kwargs)
        idx += 1

    result_array = np.mean(data_array, axis=1)
    return tested_n, result_array


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


def euclidean_distribution_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def weighted_distribution_distance(prediction, validation):
    total = np.sum(np.abs(validation))
    

def cross_val_run(fish_fnames, n, output_image=False):
    from gaussian_process_regression import rbf_regression_over_large_array
    from kernel_density_estimation import gaussian_kernel_density_estimation
    from image_processing import array_crops, export_tif_file, import_tif_file
    ffish_crops = array_crops(*import_tif_file(*fish_fnames))
    fish_shape = shape_from_slice(*ffish_crops[0][1:])
    middle_z = [(i[0].stop - i[0].start)//2 for i in ffish_crops]
    output = np.empty((n, 2))
    for i in range(n):
        splits1, splits2 = random_indices(2, len(fish_fnames))

        agglomerated_fish = np.zeros(fish_shape)
        for j in splits1:
            agglomerated_fish += import_tif_file(fish_fnames[j], dtype=float, key=middle_z[j])[ffish_crops[j][1:]]
        agglomerated_fish /= len(splits1)
        gpr = rbf_regression_over_large_array(agglomerated_fish, 0.05, 20, step=(1, 1))[0]
        kde = gaussian_kernel_density_estimation(agglomerated_fish, (20, 20))

        agglomerated_fish = np.zeros(fish_shape)
        for j in splits2:
            agglomerated_fish += import_tif_file(fish_fnames[j], dtype=float, key=middle_z[j])[ffish_crops[j][1:]]
        agglomerated_fish /= len(splits2)
        if output_image:
            if i < 4:
                export_tif_file("figures/run{}_gpr".format(i), gpr)
                export_tif_file("figures/run{}_kde".format(i), kde)

        output[i, 0] = norm_1(gpr, agglomerated_fish)
        output[i, 1] = norm_1(kde, agglomerated_fish)

    return output


def optimize_parameters(fish_fnames, func, var_kwargs, const_kwargs=None, iterations=10, output=False):
    from image_processing import array_crops, shape_from_slice, import_tif_file, export_tif_file
    from tqdm import tqdm
    if const_kwargs is None:
        const_kwargs = {}
    ordered_keys = list(var_kwargs.keys())
    values = list(var_kwargs.values())
    grid = Grid(values)
    param_score = np.empty(len(grid))
    ffish_crops = array_crops(*import_tif_file(*fish_fnames, ))
    ffish_shape = shape_from_slice(*ffish_crops[0][1:])
    middle_z = [(i[0].stop - i[0].start) // 2 for i in ffish_crops]
    num = 0
    lst = grid.get_list()
    for i in tqdm(lst):
        kwargs_i = {ordered_keys[j]: i[j] for j in range(grid.ndim)}
        kwargs_i.update(const_kwargs)
        for j in range(iterations):
            splits1, splits2 = random_indices(2, len(fish_fnames))
            agglomerated_fish = np.zeros(ffish_shape)
            for k in splits1:
                agglomerated_fish += import_tif_file(fish_fnames[k], key=middle_z[k])[ffish_crops[k][1:]]
            agglomerated_fish /= len(splits1)
            prediction = func(agglomerated_fish, **kwargs_i)
            if output and j==0:
                export_tif_file(f"figures/optimization_{func.__name__}({num})", prediction)

            agglomerated_fish = np.zeros(ffish_shape)
            for k in splits2:
                agglomerated_fish += import_tif_file(fish_fnames[k], key=middle_z[k])[ffish_crops[k][1:]]
            agglomerated_fish /= len(splits2)

            param_score[num] += norm_1(prediction, agglomerated_fish)
        num += 1
    param_score /= iterations
    dtype = [(key, var_kwargs[key].dtype) for key in ordered_keys]
    dtype.append(("score", param_score.dtype))
    dtype = np.dtype(dtype)
    lst = np.hstack((lst, exactly_2d(param_score)))
    return np.array([tuple(i) for i in lst[:]], dtype=dtype)


def average_arrays(array, ratio, threshold, threshold1=None):
    masked = np.ma.array(array, fill_value=0.)
    rng = np.random.default_rng()
    if threshold1 is not None:
        np.ma.masked_greater(masked, threshold1, copy=False)

    indices_above_threshold = np.ma.nonzero(masked > threshold)
    index_num = len(indices_above_threshold[0])
    indices = tuple(
        dim[
            rng.choice(index_num, size=int(ratio*index_num), replace=False)
        ] for dim in indices_above_threshold
    )
    masked[indices] = np.ma.masked
    output = np.zeros_like(array)
    index_array = np.ma.getmaskarray(masked)
    output[index_array] = array[index_array]
    return output
