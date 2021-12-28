from skimage import io
from numpy import dtype as _dtype


def import_tif_file(fname, dtype=_dtype(float), **kwargs):
    return io.imread(fname=fname, **kwargs).astype(dtype)

