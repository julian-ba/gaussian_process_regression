import numpy as np


def import_tif_file(*file_name: str, dtype=None, zoom=False, **kwargs) -> np.ndarray | tuple:
    from tifffile import imread
    if len(file_name) == 1:
        image = imread(*file_name, **kwargs)
        if zoom:
            from scipy.ndimage import zoom as scipy_zoom
            image = scipy_zoom(image, zoom)
        if dtype is None:
            return image
        else:
            return image.astype(dtype)
    else:
        image = [imread(file_name_i, **kwargs) for file_name_i in file_name]
        if zoom:
            from scipy.ndimage import zoom as scipy_zoom
            image = [scipy_zoom(image_i, zoom) for image_i in image]
        if dtype is None:
            return tuple(image_i.astype(dtype) for image_i in image)
        else:
            return tuple(image)


def export_tif_file(file_name: str, array: np.ndarray, dtype=None, **kwargs) -> None:
    from tifffile import imwrite

    if dtype is None:
        dtype = array.dtype

    imwrite(file_name + ".tif", array.astype(dtype), **kwargs)


def find_minimal_shape(*array) -> tuple:
    shapes = np.stack([np.array(_array.shape) for _array in array])
    minimal_shape = np.amin(shapes, axis=0)
    return tuple(minimal_shape)


def array_crops(*array) -> tuple:
    shape = np.array(find_minimal_shape(*array))
    shapes = [i.shape for i in array]
    image_slices = []
    for _shape in shapes:
        q, r = np.divmod(np.array(_shape) - shape, 2, dtype=int)
        image_slices.append(tuple(slice(q[i], _shape[i]-q[i]-r[i]) for i in range(len(q))))
    return tuple(image_slices)
