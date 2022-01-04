from skimage import io
from numpy import dtype


def import_tif_file(fname, datatype=dtype(float), **kwargs):
    return io.imread(fname=fname, **kwargs).astype(dtype=datatype)

