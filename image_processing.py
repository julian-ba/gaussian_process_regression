from core import *
from numpy import dtype


def import_tif_file(fname, datatype=dtype(float), **kwargs):
    from skimage import io

    if dtype is None:
        return io.imread(fname=fname, **kwargs)
    else:
        return io.imread(fname=fname, **kwargs).astype(dtype=datatype)


def export_tif_file(fname, array, datatype=dtype("uint8"), fit=False, **kwargs):
    from skimage import io

    if datatype is None:
        datatype = array.dtype

    if fit:
        array *= np.finfo(datatype).max / np.amax(array)

    io.imsave(fname, array.astype(datatype), **kwargs)


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

    output = []

    for i in indices:
        output.append(tuple(slice(index_lookup_table[j][i[j]-1], index_lookup_table[j][i[j]]) for j in range(ndim)))

    return output


def shape_from_slice(*slice_i):
    return tuple(i.stop - i.start for i in slice_i)


def considered_part_slices(list_of_tuples_of_slices, epsilon, upper_bounds):
    epsilon = np.broadcast_to(epsilon, (len(list_of_tuples_of_slices[0]),))
    list_of_considered_part = []
    for i in list_of_tuples_of_slices:
        list_of_new_slices = []
        for j in range(len(i)):
            start = i[j].start - epsilon[j]
            if start < 0:
                start = 0

            stop = i[j].stop + epsilon[j]
            if stop > upper_bounds[j]:
                stop = upper_bounds[j]

            list_of_new_slices.append(slice(start, stop))
        list_of_considered_part.append(tuple(list_of_new_slices))

    return list_of_considered_part


def subdivided_array_and_considered_part_slices(array, step_size, epsilon):
    sas = subdivided_array_slices(array, step_size)
    cps = considered_part_slices(sas, epsilon, array.shape)
    concatenated = [(sas[i], cps[i]) for i in range(len(sas))]
    return concatenated
