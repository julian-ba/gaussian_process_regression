from core import *
import numpy as np
import simulated_data as sd
import gaussian_process_regression as gpf


def time_model_1d(
        repetitions, x, fx=sd.smooth_data, return_seconds=False, model_type=gpf.rbf_regression_model, **kwargs
):
    import time
    # Create repetitions-# of model_type of data x and return the time in ns or s taken for each resp. repetition.
    model_type(x=np.array([[0]]), fx=np.array([[1]]))  # Should start TensorFlow

    testing_points = np.linspace(0, 10, 11)

    output_array = np.empty(repetitions)
    if callable(fx):
        n = x.shape[0]
        for repetition_idx in range(repetitions):
            data = exactly_2d(fx(n=n, **kwargs))
            t1 = time.time_ns()
            m = model_type(x=x, fx=data, **kwargs)
            m.predict_f(testing_points)
            t2 = time.time_ns()
            del m
            output_array[repetition_idx] = t2 - t1
    else:
        for repetition_idx in range(repetitions):
            t1 = time.thread_time_ns()
            m = model_type(x=x, fx=fx, **kwargs)
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
    internal_array = array
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


def euclidean_distribution_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def cross_val_run(fish, n):
    from gaussian_process_regression import rbf_regression_over_large_array
    from kernel_density_estimation import gaussian_kernel_density_estimation
    from image_processing import crop_arrays_to_same_shape
    ffish = crop_arrays_to_same_shape(*fish)
    fish_shape = ffish[0].shape
    output = np.empty((n, 2))
    for i in range(n):
        splits1, splits2 = random_indices(2, len(ffish))

        agglomerated_fish1 = np.zeros(fish_shape)
        for j in splits1:
            agglomerated_fish1 += ffish[j]
        agglomerated_fish1 /= len(splits1)
        gpr1 = rbf_regression_over_large_array(agglomerated_fish1, 0.05, 1., 1., step=(1, 3, 3))
        kde1 = gaussian_kernel_density_estimation(agglomerated_fish1, (1, 3, 3))
        print(np.nonzero(gpr1))

        agglomerated_fish2 = np.zeros(fish_shape)
        for j in splits2:
            agglomerated_fish2 += ffish[j]
        agglomerated_fish2 /= len(splits2)
        gpr2 = rbf_regression_over_large_array(agglomerated_fish2, 0.05, 1., 1., step=(1, 3, 3))
        kde2 = gaussian_kernel_density_estimation(agglomerated_fish2, (1., 3., 3.))
        print(np.nonzero(gpr2))

        output[i, 0] = euclidean_distribution_distance(gpr1, gpr2)
        output[i, 1] = euclidean_distribution_distance(kde1, kde2)

    return output


