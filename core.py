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


def coord_list_and_meshgrid(*xi):
    meshxi = np.meshgrid(*xi, indexing="ij")
    coord_grid = np.reshape(np.stack(meshxi, axis=-1), (-1, len(xi)), order="C")
    return coord_grid, meshxi


def coord_list(*xi):
    return coord_list_and_meshgrid(*xi)[0]


def sparsify(fx, threshold, *xis):
    # From an array, return a list of coordinates, and a list of values corresponding to the coordinates.
    indices = np.nonzero(fx >= threshold)
    sparse_fx = atleast_column(fx[indices])
    mgrid_specifier = tuple([slice(0, i) for i in fx.shape])
    if xis == ():
        ndim = fx.ndim
        permutation = list(range(1, ndim+1))
        permutation.append(0)
        sparse_coords = np.mgrid[mgrid_specifier].transpose(permutation)[indices]
    else:
        sparse_coords = np.stack(np.meshgrid(*xis, indexing="ij"), axis=-1)[indices]

    return sparse_coords, sparse_fx

