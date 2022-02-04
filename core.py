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


def subdivided_array_slices(array, step_size):
    ndim = array.ndim
    shape = array.shape
    step_size = np.broadcast_to(step_size, ndim)

    r = np.remainder(np.array(shape), step_size, dtype=int)

    index_lookup_table = []

    index_lookup_table_bounds_minus_1 = []

    for i in range(ndim):
        index_lookup_table.append(np.arange(0, shape[i], step_size[i], dtype=int))
        if r[i] != 0:
            index_lookup_table[i] = np.append(index_lookup_table[i], shape[i])

        index_lookup_table_bounds_minus_1.append(slice(0, len(index_lookup_table[i]) - 1))

    indices = Grid(index_lookup_table_bounds_minus_1).get_list() + 1

    output = []

    for i in indices:
        output.append(tuple(slice(index_lookup_table[j][i[j]-1], index_lookup_table[j][i[j]]) for j in range(ndim)))


    return output


def shape_from_slice(*slice_i):
    return tuple(i.stop - i.start for i in slice_i)


def considered_part_slices(list_of_tuples_of_slices, epsilon, upper_bounds):
    epsilon = np.broadcast_to(epsilon, (len(list_of_tuples_of_slices[0]),)).astype(int)
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


def sparsify(
        fx, threshold, slices=None, step=None, dtype_sparse_coords=np.dtype(float), dtype_sparse_fx=np.dtype(float)
):
    # From an array, return a list of coordinates, and a list of values corresponding to the coordinates, where all the
    # values are greater than threshold.
    indices = np.nonzero(fx >= threshold)
    sparse_fx = exactly_2d(fx[indices])
    if slices is None:
        grid = Grid(fx.shape, step)
    else:
        grid = Grid(slices, step)

    coordinate_array = grid.get_array()
    sparse_coords = coordinate_array[indices]

    if dtype_sparse_coords is not None:
        sparse_coords = sparse_coords.astype(dtype_sparse_coords)

    if dtype_sparse_fx is not None:
        sparse_fx = sparse_fx.astype(dtype_sparse_fx)

    return sparse_coords, sparse_fx


class LargeArrayIterator:
    # Currently not functional. Do not use.
    def __init__(self, array, n, threshold, eps, step):
        from numpy import ma
        self.array = ma.masked_array(array, copy=True)
        self.ndim = self.array.ndim
        self.shape = self.array.shape
        self.n = n
        self.step = step
        #  self.integer_step =
        self.threshold = threshold
        self.eps = np.array(eps)

    def __iter__(self):
        return self

    def __next__(self):
        from numpy import ma
        if ma.count(self.array) == 0:
            raise StopIteration
        else:
            first_not_evaluated_point = np.argwhere(ma.getmaskarray(self.array))[0]


class Grid:
    def convert_to_coords(self, step):
        step = np.broadcast_to(step, (self.ndim,))
        for i in range(self.ndim):
            self.axes[i] *= step[i]
        self.indexable = False

    def _initialize_from_shape(self, shape):
        self.shape = shape
        slices = [slice(i) for i in self.shape]
        self._initialize_from_slices(slices)

    def _initialize_from_slices(self, slices):
        list_of_kwargs = []
        for slice_ in slices:
            kwargs = {}
            if slice_.start is not None:
                kwargs["start"] = slice_.start
            if slice_.stop is not None:
                kwargs["stop"] = slice_.stop
            if slice_.step is not None:
                kwargs["step"] = slice_.step
            list_of_kwargs.append(kwargs)
        self.axes = [np.arange(**kwargs) for kwargs in list_of_kwargs]
        self.slices = list(slices)
        self.ndim = len(self.axes)

    def __init__(self, init, step=None):
        if all([isinstance(i, slice) for i in init]):
            self._initialize_from_slices(init)
            self.shape = tuple([len(axis) for axis in self.axes])

        elif all([isinstance(i, int) for i in init]):
            init = tuple(init)
            self._initialize_from_shape(init)

        elif isinstance(init, np.ndarray):
            self._initialize_from_shape(init.shape)

        else:
            raise ValueError

        self.indexable = True
        self.size = np.prod(self.shape)

        if step is not None:
            self.convert_to_coords(step)

    def get_array(self, dtype=None):
        if self.indexable:
            cyclic_permutation = list(range(1, self.ndim + 1))
            cyclic_permutation.append(0)
            out = np.mgrid[self.slices].transpose(cyclic_permutation)

        else:
            out = np.stack(np.meshgrid(*self.axes, indexing="ij"), axis=-1)

        if dtype is not None:
            return out.astype(dtype)
        else:
            return out

    def get_list(self, dtype=None):
        return self.get_array(dtype).reshape((-1, self.ndim))

    def get_axis(self, *i, dtype=None):
        if dtype is not None:
            return tuple(self.axes[idx].astype(dtype) for idx in i)
        else:
            return tuple(self.axes[idx] for idx in i)

    def to_array(self, points):
        points = exactly_2d(np.array(points))
        assert points.size == self.size
        return points.reshape(self.shape)

