# Convenience functions
import numpy as np


def exactly_2d(x):
    # Warning: This function does not necessarily behave as expected with arrays of dimension greater than 2. In this
    # case, it should only be used carefully.
    x = np.asarray(x)
    if x.ndim <= 1:
        output_array = np.atleast_2d(x).T
    elif x.ndim == 2:
        output_array = x
    else:
        output_array = x.reshape((-1, x.shape[-1]))
    return output_array


def coord_array(*xi):
    return np.stack(np.meshgrid(*xi, indexing="ij"), axis=-1)


def index_array(*slices):
    ndim = len(slices)
    cyclic_permutation = list(range(1, ndim + 1))
    cyclic_permutation.append(0)
    return np.mgrid[slices].transpose(cyclic_permutation)


def coord_or_index_array(*xi_or_slice):
    if all((isinstance(i, slice) for i in xi_or_slice)):
        return index_array(*xi_or_slice)
    elif all((isinstance(i, np.ndarray) for i in xi_or_slice)):
        return coord_array(*xi_or_slice)
    else:
        raise ValueError("The only allowed arguments are tuples of NumPy arrays or slices.")


def coord_or_index_list(*xi_or_slice):
    return coord_or_index_array(*xi_or_slice).reshape((-1, len(xi_or_slice)))


def sparsify(fx, threshold, *xi_or_slice, dtype_sparse_coords=np.dtype(float), dtype_sparse_fx=np.dtype(float)):
    # From an array, return a list of coordinates, and a list of values corresponding to the coordinates, where all the
    # values are greater than threshold.
    indices = np.nonzero(fx >= threshold)
    sparse_fx = exactly_2d(fx[indices])
    if xi_or_slice == ():
        xi_or_slice = (slice(0, i) for i in fx.shape)

    coordinate_array = coord_or_index_array(*xi_or_slice)
    sparse_coords = coordinate_array[indices]

    if dtype_sparse_coords is not None:
        sparse_coords = sparse_coords.astype(dtype_sparse_coords)

    if dtype_sparse_fx is not None:
        sparse_fx = sparse_fx.astype(dtype_sparse_fx)

    return sparse_coords, sparse_fx


def subdivided_array_slices(array, step_size=10):
    ndim = array.ndim
    shape = array.shape
    step_size, shape = np.broadcast_arrays(step_size, shape)

    q, r = np.divmod(shape, step_size)

    index_lookup_table = []

    index_lookup_table_bounds_minus_1 = []

    for i in range(ndim):
        index_lookup_table.append(np.arange(q[i]+1) * step_size[i])
        if r[i] != 0:
            index_lookup_table[i] = np.append(index_lookup_table[i], shape[i])

        index_lookup_table_bounds_minus_1.append(slice(0, len(index_lookup_table[i]) - 1))

    indices = coord_or_index_list(*index_lookup_table_bounds_minus_1) + 1

    return (tuple(slice(index_lookup_table[j][i[j]-1], index_lookup_table[j][i[j]]) for j in range(ndim)) for i in indices)


def euclidean_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def euclidean_norm(x):
    return euclidean_distance(x, x)


def to_radial_function(function):

    def radial_function(x):
        return function(euclidean_norm(x))

    return radial_function


