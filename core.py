# Convenience functions
import numpy as np
import numbers


def exactly_2d(x):
    if x.ndim <= 1:
        output_array = np.atleast_2d(x).T
    elif x.ndim == 2:
        output_array = x
    else:
        dim = x.shape[1]
        output_array = np.empty_like(x).reshape(-1, dim)
        for j in range(dim):
            output_array[:, j] = x[:, j, ...].flatten()
    return output_array


def coord_list_and_meshgrid(*xi):
    meshxi = np.meshgrid(*xi, indexing="ij")
    coord_grid = np.reshape(np.stack(meshxi, axis=-1), (-1, len(xi)), order="C")
    return coord_grid, meshxi


def coord_list(*xi):
    return coord_list_and_meshgrid(*xi)[0]


def coord_list_evenly_spaced(*upper_bound):
    xi = [np.arange(i) for i in upper_bound]
    return coord_list(*xi)


def sparsify(fx, threshold, dtype_sparse_coords=np.dtype(float), dtype_sparse_fx=np.dtype(float), *xis):
    # From an array, return a list of coordinates, and a list of values corresponding to the coordinates, where all the
    # values are greater than threshold.
    indices = np.nonzero(fx >= threshold)
    sparse_fx = exactly_2d(fx[indices])
    mgrid_specifier = tuple([slice(0, i) for i in fx.shape])
    if xis == ():
        ndim = fx.ndim
        cyclic_permutation = list(range(1, ndim+1))
        cyclic_permutation.append(0)
        sparse_coords = np.mgrid[mgrid_specifier].transpose(cyclic_permutation)[indices]
    else:
        sparse_coords = np.stack(np.meshgrid(*xis, indexing="ij"), axis=-1)[indices]

    if dtype_sparse_coords is not None:
        sparse_coords = sparse_coords.astype(dtype_sparse_coords)

    if dtype_sparse_fx is not None:
        sparse_fx = sparse_fx.astype(dtype_sparse_fx)

    return sparse_coords, sparse_fx

