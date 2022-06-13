# Convenience functions
import numpy as np

def _exactly_2d(array):
    if array.ndim <= 1:
        return array[:, np.newaxis]
    elif array.ndim == 2:
        return array
    else:
        return array.reshape((-1, array.shape[-1]))


def exactly_2d(*array: np.ndarray) -> np.ndarray | tuple:
    if len(array) == 1:
        return _exactly_2d(array[0])
    else:
        return tuple(_exactly_2d(array_i) for array_i in array)


def find_minimal_shape(*array) -> tuple:
    shapes = np.stack([np.array(_array.shape) for _array in array])
    minimal_shape = np.amin(shapes, axis=0)
    return tuple(minimal_shape)


def array_crops(*array, shape=None) -> tuple:
    if shape is not None:
        minimal_shape = np.array(find_minimal_shape(*array, np.empty(shape)))
    else:
        minimal_shape = np.array(find_minimal_shape(*array))
    shapes = [i.shape for i in array]
    image_slices = []
    for _shape in shapes:
        q, r = np.divmod(np.array(_shape) - minimal_shape, 2, dtype=int)
        image_slices.append(tuple(slice(q[i], _shape[i]-q[i]-r[i]) for i in range(len(q))))
    return tuple(image_slices)


class Grid:
    def convert_to_coordinates(self, step=None):
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

    def __init__(self, *init: slice | tuple | np.ndarray | int, step: None | tuple=None):
        self.axes = None
        self.slices = None
        self.indexable = True
        self.size = None
        self.ndim = None
        self.shape = None

        if all([isinstance(i, slice) for i in init]):
            self._initialize_from_slices(init)

        elif all([isinstance(i, tuple) for i in init]) and len(init) == 1:
            self._initialize_from_shape(init[0])

        elif all([isinstance(i, int) for i in init]):
            self._initialize_from_shape(init)

        elif all([isinstance(i, np.ndarray) for i in init]) and len(init) == 1:
            self._initialize_from_shape(init[0].shape)

        elif all([isinstance(i, np.ndarray) for i in init]):
            self._initialize_from_axes(init)

        else:
            raise ValueError

        self._update_attributes()

        if step is not None:
            self.convert_to_coordinates(step)

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

    @property
    def array(self):
        return self.get_array()

    def get_list(self, dtype=None):
        return self.get_array(dtype=dtype).reshape((-1, self.ndim))

    @property
    def list(self):
        return self.get_list()

    def __getitem__(self, item: int | slice | tuple):
        from warnings import warn
        warn("Prefer directly calling the list or array methods of Grid object.")
        if isinstance(item, int) or isinstance(item, slice):
            return self.list[item]
        elif isinstance(item, tuple):
            return self.array[item]
        else:
            raise ValueError(
                "Item must be int, slice or tuple. If this function does not work correctly, "+
                "try explicitly calling the array or list methods."
            )

    def __iter__(self):
        self._list = self.list  # Saves self._list to avoid repeated calling of Grid.list
        self._num = 0
        return self

    def __next__(self):
        if self._num == self.size:
            del self._list
            del self._num
            raise StopIteration
        else:
            old_num = self._num
            self._num += 1
            return self._list[old_num]


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

    @staticmethod
    def from_points_to_array(shape, *point_list):
        if len(point_list) == 1:
            point_list = exactly_2d(*point_list)
            return point_list.reshape(shape)
        else:
            point_list = exactly_2d(*point_list)
            return tuple(points_i.reshape(shape) for points_i in point_list)

    def to_array(self, *point_list):
        return self.from_points_to_array(self.shape, *point_list)


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
        *array:np.ndarray,
        threshold=0.05, slices: None | tuple=None, step: None | tuple=None, include_zeros:bool=True,
        dtype_sparse_coordinates=np.dtype(float), dtype_sparse_fx=np.dtype(float)
):
    # From an array, return a list of coordinates, and a list of values corresponding to the coordinates, where all the
    # values are greater than threshold.
    assert isinstance(include_zeros, bool), "include_zeros MUST be bool."
    if include_zeros:
        indices = np.nonzero(np.any(np.stack(array) >= threshold, axis=0))
        sparse_fx = np.concatenate([array_i[indices] for array_i in array])
        if dtype_sparse_fx is not None:
            sparse_fx = sparse_fx.astype(dtype_sparse_fx)

        if slices is None:
            grid = Grid(array[0].shape, step=step)
        else:
            grid = Grid(*slices, step=step)

        sparse_coords = np.concatenate([grid.get_array(dtype_sparse_coordinates)[indices]] * len(array))
        return sparse_coords, sparse_fx
    else:
        indices = tuple(np.nonzero(array_i >= threshold) for array_i in array)
        sparse_fx = np.concatenate(tuple(exactly_2d(array[idx][indices[idx]]) for idx in range(len(array))))
        if slices is None:
            grids = tuple(Grid(array[i].shape, step=step) for i in range(len(array)))
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
        epsilon = np.array(epsilon)
        lower_evaluate = np.broadcast_to(lower_evaluate, ndim_shape).astype(int)
        if upper_evaluate is None:
            step = np.array(step)
            upper_evaluate = np.minimum(lower_evaluate + step, shape, dtype=int)
        else:
            upper_evaluate = np.broadcast_to(upper_evaluate, ndim_shape).astype(int )

        self.consider = None
        if np.any(epsilon) or upper_consider is not None or lower_consider is not None:
            if lower_consider is None:
                lower_consider = np.maximum(lower_evaluate - epsilon, 0)
            lower_consider = np.broadcast_to(lower_consider, ndim_shape).astype(int)

            if upper_consider is None:
                upper_consider = np.minimum(upper_evaluate + epsilon, shape)
            upper_consider = np.broadcast_to(upper_consider, ndim_shape).astype(int)
            self.consider = tuple(slice(lower_consider[dim], upper_consider[dim]) for dim in range(ndim))

        self.evaluate = tuple(slice(lower_evaluate[dim], upper_evaluate[dim]) for dim in range(ndim))

    def __repr__(self):
        return f"AnalysisSlices(evaluate: {repr(self.evaluate)}; consider: {repr(self.consider)})"


class LargeArrayIterator:
    def __init__(self, array, grid_shape=None, epsilon:int=0, method:str="grid"):
        shape = array.shape

        if method == "grid":
            index_lookup_table = []
            if grid_shape is None:
                grid_shape = tuple(np.ceil(np.array(array.shape) / 10).astype(np.dtype(int)))
            assert np.all(grid_shape)
            grid_shape = np.broadcast_to(grid_shape, array.ndim)
            for dim in range(array.ndim):
                index_lookup_table.append(
                    np.append(np.arange(0, shape[dim] - 1, grid_shape[dim], dtype=int), shape[dim]))
            self.indices_lower = Grid(*[coords[:-1] for coords in index_lookup_table]).get_list()
            self.indices_upper = Grid(*[coords[1:] for coords in index_lookup_table]).get_list()

            if np.any(epsilon):
                self.considered_lower = np.maximum(self.indices_lower - epsilon, 0)
                self.considered_upper = np.minimum(self.indices_upper + epsilon, shape)
            else:
                self.considered_lower = [None] * len(self.indices_lower)
                self.considered_upper = [None] * len(self.indices_lower)

        else:
            raise NotImplementedError("Currently, only grid covering is implemented.")

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

    def __getitem__(self, item:int):
        if item > len(self):
            raise ValueError
        return _AnalysisSlices(
            lower_evaluate=self.indices_lower[item],
            upper_evaluate=self.indices_upper[item],
            lower_consider=self.considered_lower[item],
            upper_consider=self.considered_upper[item]
        )
