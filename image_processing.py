import numpy as np
from user_values import DATA_DIRECTORY
from os import listdir, path


def import_tif_files(directory: str = DATA_DIRECTORY, dtype=None, zoom=False, **kwargs) -> tuple:
    from tifffile import imread
    from scipy.ndimage import zoom as scipy_zoom

    images_filenames = [path.join(directory, filename_i) for filename_i in listdir(path=directory)]
    if len(images_filenames) == 1:
        image = imread(images_filenames[0], **kwargs)
        if zoom:
            image = scipy_zoom(image, zoom)
        if dtype is None:
            return image.astype(dtype)
        else:
            return image
    else:
        image = [imread(filename_i, **kwargs) for filename_i in images_filenames]
        if zoom:
            image = [scipy_zoom(image_i, zoom) for image_i in image]
        if dtype is None:
            return tuple(image_i.astype(dtype) for image_i in image)
        else:
            return tuple(image)


def export_tif_file(file_name: str, array: np.ndarray, dtype=None, **kwargs) -> None:
    from tifffile import imwrite

    if dtype is None:
        dtype = array.dtype

    imwrite(file_name, array.astype(dtype), **kwargs)
