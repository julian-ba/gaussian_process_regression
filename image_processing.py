from core import *
import numpy as np

FISH_FNAMES = [
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish10_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish11_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish12_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish13_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish14_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish15_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish17_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish18_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif"
]


def import_tif_file(*file_name: str, dtype=None, **kwargs):
    from tifffile import imread
    if len(file_name) == 1:
        if dtype is None:
            return imread(*file_name, **kwargs)
        else:
            return imread(*file_name, **kwargs).astype(dtype)
    else:
        if dtype is None:
            return tuple(imread(file_name_i, **kwargs) for file_name_i in file_name)
        else:
            return tuple(imread(file_name_i, **kwargs).astype(dtype) for file_name_i in file_name)


def export_tif_file(file_name: str, array: np.ndarray, dtype=None, **kwargs):
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
