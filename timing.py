from core import *
import time
import numpy as np
import simulated_data as sd


def time_model(
        repetitions, x, fx=sd.smooth_data, return_seconds=False, model_type=rbf_regression, **kwargs
):
    # Create repetitions-# of model_type of data x and return the time in ns or s taken for each resp. repetition.
    model_type(x=np.array([[0]]), fx=np.array([[1]]))  # Starts TensorFlow
    output_array = np.empty(repetitions)
    if callable(fx):
        n = x.shape[0]
        for repetition_idx in range(repetitions):
            data = atleast_column(fx(n=n, **kwargs))
            t1 = time.time_ns()
            model_type(x=x, fx=data, **kwargs)
            t2 = time.time_ns()
            output_array[repetition_idx] = t2 - t1
    else:
        for repetition_idx in range(repetitions):
            t1 = time.thread_time_ns()
            model_type(x=x, fx=fx, **kwargs)
            t2 = time.thread_time_ns()
            output_array[repetition_idx] = t2 - t1

    if return_seconds:
        return output_array * 1e-9
    else:
        return output_array


def run_1d_test(repetitions, lower_n, upper_n, step=1, *args, **kwargs):
    tested_n = np.arange(start=lower_n, stop=upper_n+1, step=step, dtype=int)
    data_array = np.empty((len(tested_n), repetitions))
    idx = 0
    for i in tested_n:
        x = np.linspace(0, 10, i)
        data_array[idx] = time_model(repetitions=repetitions, x=x, *args, **kwargs)
        idx += 1

    result_array = np.mean(data_array, axis=1)
    return tested_n, result_array
