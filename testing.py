from core import *
import numpy as np
import simulated_data as sd
import gaussian_process_regression as gpf


def time_model_1d(
        repetitions, x, fx=sd.smooth_data, return_seconds=False, model_type=gpf.rbf_regression, **kwargs
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


def generate_random_indices(number_of_splits, indices):
    # Probably could be handled more efficiently
    indices = np.array(indices)
    q, r = np.divmod(len(indices), number_of_splits)[j]
    list_of_splits = []
    for i in range(number_of_splits):
        indices_array = np.empty(q + min(max(0, r), 1))
        with np.nditer(indices_array, op_flags=["write"]) as it:
            for j in it:
                index_to_pop = np.random.randint(0, len(indices))
                j = indices[indices_array]
                indices = np.delete(indices, index_to_pop)
        list_of_splits.append(indices_array)
        r -= 1

    return tuple(list_of_splits)


def distribution_distance(x, y, fx, fy, method):


def cross_validation_run(x, fx, n, *methods):
    output = np.empty((n, len(methods)))
    splits = tuple(generate_random_indices(2, len(x)) for i in range(n))

    with np.nditer(output, flags=["multi_index"], op_flags=["write"]) as it:
        for i in it:
            for j in splits[it.index[0]]:



