import numpy as np
from typing import Tuple
import pathlib


def import_tif_file(file: str, dtype=None, zoom=False, **kwargs) -> np.ndarray:
    from tifffile import imread
    from scipy.ndimage import zoom as scipy_zoom

    image = imread(file, **kwargs)
    if zoom:
        image = scipy_zoom(image, zoom)
    if dtype is None:
        return image.astype(dtype)
    else:
        return image


def export_tif_file(file_name: str, array: np.ndarray, dtype=None, **kwargs) -> None:
    from tifffile import imwrite

    if dtype is None:
        dtype = array.dtype

    imwrite(file_name, array.astype(dtype), **kwargs)


def get_tif_file_names(directory: str = "") -> Tuple[str, ...]:
    paths = list((pathlib.Path("./data") / directory).glob("*.tif"))
    return tuple(str(path) for path in paths)


def import_if_str(array: np.ndarray | str, **kwargs):
    if isinstance(array, str):
        return import_tif_file(file=array, **kwargs)
    elif isinstance(array, np.ndarray):
        return array
    else:
        raise ValueError("The parameter \"array\" must be a string containing the path of a TIF file or a NumPy array.")

