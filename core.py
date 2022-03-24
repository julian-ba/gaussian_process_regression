# Convenience functions
import numpy as np
GRID_STEP_3D = (8., 0.406, 0.406)  # Step of testing data in microns, in order (delta z, delta x, delta y)
GRID_STEP_2D = (0.406, 0.406)
THRESHOLD = 0.05


def exactly_2d(x: np.ndarray) -> np.ndarray:
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


class Grid:
    def convert_to_coordinates(self, step=None, smallest=None):
        if step is None:
            step = 1.
        step = np.broadcast_to(step, (self.ndim,))
        self.axes = []
        for idx in range(self.ndim):
            slice_ = self.slices[idx]
            kwargs = {}
            if slice_.start is not None:
                kwargs["start"] = slice_.start
            if slice_.stop is not None:
                kwargs["stop"] = slice_.stop
            if slice_.step is not None:
                kwargs["step"] = slice_.step
            self.axes.append(np.arange(**kwargs)*step[idx])
        self.indexable = False
        self.slices = None
        if smallest is not None:
            smallest = np.broadcast_to(step, (self.ndim,))
            for i in range(self.ndim):
                self.axes[i] += smallest[i]
        self._update_attributes()

    def _initialize_from_shape(self, shape):
        slices = [slice(i) for i in shape]
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
        self.slices = list(slices)

    def _initialize_from_axes(self, axes):
        self.axes = axes
        self.indexable = False

    def _update_attributes(self):
        if self.indexable:
            self.ndim = len(self.slices)
            self.shape = []
            for slice_ in self.slices:
                if slice_.step is None:
                    step = 1
                else:
                    step = slice_.step
                if slice_.start is None:
                    start = 0
                else:
                    start = slice_.start
                self.shape.append((slice_.stop - start) // step)
            self.slices = tuple(self.slices)
        else:
            self.ndim = len(self.axes)
            self.shape = tuple(len(axis) for axis in self.axes)
        self.size = np.prod(self.shape)

    def __init__(self, *init, step=None, smallest=None):
        self.axes = None
        self.slices = None
        self.indexable = True
        self.size = None
        self.ndim = None
        self.shape = None

        if all([isinstance(i, slice) for i in init]):
            self._initialize_from_slices(init)

        elif all([isinstance(i, tuple) for i in init]):
            init = init[0]
            self._initialize_from_shape(init)

        elif all([isinstance(i, np.ndarray) for i in init]) and len(init) == 1:
            self._initialize_from_shape(init[0].shape)

        elif all([isinstance(i, np.ndarray) for i in init]):
            self._initialize_from_axes(init)

        else:
            raise ValueError

        self._update_attributes()

        if step is not None or smallest is not None:
            self.convert_to_coordinates(step, smallest)

    def __len__(self):
        return self.size

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
        if self.indexable:
            if len(i) == 1:
                return np.arange(self.slices[i[0]].start, self.slices[i[0]].stop, self.slices[i[0]].step)
            else:
                return tuple(
                    np.arange(self.slices[idx].start, self.slices[idx].stop, self.slices[idx].step) for idx in i
                )
        else:
            if dtype is not None:
                if len(i) == 1:
                    self.axes[i[0]].astype(dtype)
                else:
                    return tuple(self.axes[idx].astype(dtype) for idx in i)
            else:
                if len(i) == 1:
                    return self.axes[i[0]]
                else:
                    return tuple(self.axes[idx] for idx in i)

    def to_array(self, points):
        points = exactly_2d(np.array(points))
        assert points.size == self.size
        return points.reshape(self.shape)

    @staticmethod
    def from_points_to_array(shape, points):
        points = exactly_2d(np.array(points))
        return points.reshape(shape)


def get_ndim(*args):
    from numbers import Number
    dim = 0
    for arg in args:
        arg_dim = 0
        if hasattr(arg, "__len__"):
            arg_dim = len(arg)
        elif isinstance(arg, Number):
            arg_dim = 1
        dim = max(arg_dim, dim)
    return dim


def shape_from_slice(*slice_i:slice) -> tuple:
    return tuple(i.stop - i.start for i in slice_i)


def sparsify(
        *array:np.ndarray, threshold=0.05, slices=None, step=None, dtype_sparse_coordinates=np.dtype(float), dtype_sparse_fx=np.dtype(float)
):
    # From an array, return a list of coordinates, and a list of values corresponding to the coordinates, where all the
    # values are greater than threshold.
    indices = tuple(np.nonzero(array_i >= threshold) for array_i in array)
    sparse_fx = np.concatenate(tuple(exactly_2d(array[idx][indices[idx]]) for idx in range(len(array))))
    if slices is None:
        grids = (Grid(array[i].shape, step=step) for i in range(len(array)))
    else:
        grids = [Grid(*slices, step=step)] * len(array)

    sparse_coords = np.concatenate(tuple(grids[idx].get_array()[indices[idx]] for idx in range(len(grids))))

    if dtype_sparse_coordinates is not None:
        sparse_coords = sparse_coords.astype(dtype_sparse_coordinates)

    if dtype_sparse_fx is not None:
        sparse_fx = sparse_fx.astype(dtype_sparse_fx)

    return sparse_coords, sparse_fx


class _AnalysisSlices:
    def __init__(
            self, lower_evaluate,
            upper_evaluate=None, shape=None, lower_consider=None, upper_consider=None, step=None, epsilon:int=0
    ):
        ndim = get_ndim(lower_evaluate, upper_evaluate, shape, epsilon, lower_consider, upper_consider, step)
        assert ndim > 0
        ndim_shape = (ndim,)
        if epsilon is None:
            epsilon = 0
        epsilon = np.array(epsilon)
        lower_evaluate = np.broadcast_to(lower_evaluate, ndim_shape).astype(int)
        if upper_evaluate is None:
            step = np.array(step)
            upper_evaluate = np.minimum(lower_evaluate + step, shape, dtype=int)
        else:
            upper_evaluate = np.broadcast_to(upper_evaluate, ndim_shape).astype(int )

        if lower_consider is None:
            lower_consider = np.maximum(lower_evaluate - epsilon, 0)
        lower_consider = np.broadcast_to(lower_consider, ndim_shape).astype(int)

        if upper_consider is None:
            upper_consider = np.minimum(upper_evaluate + epsilon, shape)
        upper_consider = np.broadcast_to(upper_consider, ndim_shape).astype(int)

        self.evaluate = tuple(slice(lower_evaluate[dim], upper_evaluate[dim]) for dim in range(ndim))
        self.consider = tuple(slice(lower_consider[dim], upper_consider[dim]) for dim in range(ndim))
    def __repr__(self):
        return f"AnalysisSlices(evaluate: {repr(self.evaluate)}; consider: {repr(self.consider)})"


class LargeArrayIterator:
    def __init__(self, array, step_size=None, epsilon:int=0, method="grid"):
        shape = array.shape

        if method == "grid":
            index_lookup_table = []
            if step_size is None:
                step_size = tuple(np.ceil(np.array(array.shape) / 10).astype(np.dtype(int)))
            assert np.all(step_size)
            step_size = np.broadcast_to(step_size, array.ndim)
            for dim in range(array.ndim):
                index_lookup_table.append(
                    np.append(np.arange(0, shape[dim] - 1, step_size[dim], dtype=int), shape[dim]))
            self.indices_lower = Grid(*[coords[:-1] for coords in index_lookup_table]).get_list()
            self.indices_upper = Grid(*[coords[1:] for coords in index_lookup_table]).get_list()

            self.considered_lower = np.maximum(self.indices_lower - epsilon, 0)
            self.considered_upper = np.minimum(self.indices_upper + epsilon, shape)

        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.indices_lower)

    def __iter__(self):
        self._slice_num = 0
        self._stop = len(self.indices_lower)
        return self

    def __next__(self):
        if self._slice_num == self._stop:
            raise StopIteration
        else:
            old_slice_num = self._slice_num
            self._slice_num += 1
            return _AnalysisSlices(
                lower_evaluate=self.indices_lower[old_slice_num],
                upper_evaluate=self.indices_upper[old_slice_num],
                lower_consider=self.considered_lower[old_slice_num],
                upper_consider=self.considered_upper[old_slice_num]
            )

    def __getitem__(self, item):
        assert isinstance(item, int)
        if item > len(self):
            raise ValueError
        return _AnalysisSlices(
            lower_evaluate=self.indices_lower[item],
            upper_evaluate=self.indices_upper[item],
            lower_consider=self.considered_lower[item],
            upper_consider=self.considered_upper[item]
        )
