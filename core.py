# Convenience functions
import numpy as np


def exactly_2d(x):
    # Warning: This function does not necessarily behave as expected with arrays of dimension greater than 2. In this
    # case, it should only be used carefully.
    x = np.asarray(x)
    if x.ndim <= 1:
        output_array = x[:, np.newaxis]
    elif x.ndim == 2:
        output_array = x
    else:
        output_array = x.reshape((-1, x.shape[-1]))
    return output_array


def coord_array(*xi):
    return np.stack(np.meshgrid(*xi, indexing="ij"), axis=-1)


def index_to_coord_array_via_step(array, step_sizes):
    step_sizes = np.broadcast_to(step_sizes, (array.shape[-1],))
    step_sizes = np.expand_dims(step_sizes, tuple((i for i in range(array.dim - 1))))
    return array * step_sizes


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
        raise ValueError("The only allowed arguments are tuples of exclusively NumPy arrays or slices.")


def coord_array_from_shape_and_step(shape, step):
    shape, step = np.broadcast_arrays(shape, step)
    xis = []
    for i in range(len(shape)):
        xis.append(np.arange(shape[i]) * step[i])

    return coord_or_index_array(*xis)


def index_array_from_shape(shape):
    slices = tuple((slice(0, i) for i in shape))
    return coord_or_index_array(*slices)


def coord_or_index_list(*xi_or_slice):
    return coord_or_index_array(*xi_or_slice).reshape((-1, len(xi_or_slice)))


def index_list_from_shape(shape):
    slices = tuple((slice(0, i) for i in shape))
    return coord_or_index_list(*slices)


def xis_from_shape_and_step(shape, step):
    shape, step = np.broadcast_arrays(shape, step)
    xis = []
    for i in range(len(shape)):
        xis.append(np.arange(shape[i]) * step[i])
    return tuple(xis)


def coord_list_from_shape_and_step(shape, step):
    return coord_or_index_list(*xis_from_shape_and_step(shape, step))


def to_coord_list_from_step(index_list, step):
    step = exactly_2d(np.broadcast_to(step, index_list.shape[1]))
    return index_list * step


def list_from_array(array):
    return array.reshape((-1, len(array.shape)))


def to_array_from_list_and_shape(point_list, shape):
    return point_list.reshape(shape)

