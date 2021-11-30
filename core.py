# TODO: Create function which returns a list of the coordinates of a grid.
import numpy as np
import pandas as pd


def create_dataframe(array, n_val=0, value_names=None, do_binarize=False, return_spatial_dim=False):
    # Creates a pandas DataFrame, mainly for visualization purposes.
    init_dict = {}
    arr = array

    if value_names is None or len(value_names) == 0:
        value_names = []
        for i in range(n_val):
            value_names.append("v{}".format(i))

    n_val = max(len(value_names), n_val)

    try:
        n_dim = arr.shape[1]
    except IndexError:
        n_dim = 1
        arr = arr[:, np.newaxis]
    n_spatial_dim = n_dim - n_val

    if n_spatial_dim <= 0:
        raise TypeError("The number of spatial dimensions must be between 1 and 3.")

    init_dict["x"] = arr[:, 0]
    if n_spatial_dim >= 2:
        init_dict["y"] = arr[:, 1]
        if n_spatial_dim == 3:
            init_dict["z"] = arr[:, 2]
        else:
            raise TypeError("The number of spatial dimensions must be between 1 and 3.")

    if do_binarize:
        init_dict["is_point"] = np.ones(len(arr))

    i = n_spatial_dim
    for j in value_names:
        init_dict[j] = arr[:, i]
        i += 1
    k = i
    while i < arr.shape[1]:
        init_dict["v{}".format(i-k)] = arr[:, i]  # Add the rest of the data points without passed names
        i += 1

    if return_spatial_dim:
        return pd.DataFrame(init_dict), n_spatial_dim
    else:
        return pd.DataFrame(init_dict)

