# TODO: Create function which returns a list of the coordinates of a grid.
import numpy as np


def atleast_column(x):
    try:
        x = np.asarray(x)
        if x.ndim == 1:
            output_array = x[:, np.newaxis]
        else:
            output_array = x
    except TypeError:
        raise TypeError("The input must be a NumPy-array.")

    return output_array
