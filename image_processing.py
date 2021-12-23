import numpy as np
from core import *
from skimage import io


def import_tif_file(fname, dtype=np.dtype(float), **kwargs):
    return io.imread(fname=fname, **kwargs).astype(dtype)
