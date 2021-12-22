# Convenience functions
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


# def flatten_values(x, output_dim=1):


def coord_list(*xi):
    # TODO: Verify correct functionality
    coord_grid = np.reshape(np.stack(np.meshgrid(*xi), axis=-1), (-1, len(xi)), order="C")
    return coord_grid


def coord_list_and_meshgrid(*xi):
    meshxi = np.meshgrid(*xi)
    coord_grid = np.reshape(np.stack(meshxi, axis=-1), (-1, len(xi)), order="C")
    return coord_grid, meshxi


def sparsify(fx, threshold, *xis):
    # From an array, return a list of coordinates, and a list of values corresponding to the coordinates.
    indices = np.nonzero(fx >= threshold)
    sparse_fx = atleast_column(fx[indices])
    if xis == ():
        sparse_coords = np.mgrid[indices]
    else:
        sparse_coords = np.stack(np.meshgrid(*xis), axis=-1)[indices]

    return sparse_coords, sparse_fx
