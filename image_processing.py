from core import *
from numpy import dtype

fish_fnames = [
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish10_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish11_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish12_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish13_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish14_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish15_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish17_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif",
    "C:/Users/jfb20/Desktop/test_funcMaps_fig2D_Batch_correlation_2regs_VARoption/save_fish18_fig2D_Batch_correlation_2regs_VARoption.mat_func.tif"
]


def import_tif_file(*fname, datatype=None, **kwargs):
    from skimage import io
    if len(fname) == 1:
        if datatype is None:
            return io.imread(*fname, **kwargs)
        else:
            return io.imread(*fname, **kwargs).astype(datatype)
    else:
        if datatype is None:
            return tuple(io.imread(fnamei, **kwargs) for fnamei in fname)
        else:
            return tuple(io.imread(fnamei, **kwargs).astype(datatype) for fnamei in fname)


def export_tif_file(fname, array, datatype=dtype("uint16"), fit=False, **kwargs):
    from tifffile import imwrite

    if datatype is None:
        datatype = array.dtype

    if fit:
        if np.all(array == 0):
            pass
        else:
            normalization_coefficient = np.divide(float(np.iinfo(datatype).max), np.amax(array))
            array *= normalization_coefficient

    imwrite(fname+".tif", array.astype(datatype), **kwargs)


def find_minimal_shape(*array):
    shapes = np.stack([np.array(_array.shape) for _array in array])
    minimal_shape = np.amin(shapes, axis=0)
    return minimal_shape


def array_crops(*array):
    shape = np.array(find_minimal_shape(*array))
    shapes = [i.shape for i in array]
    image_slices = []
    for _shape in shapes:
        q, r = np.divmod(np.array(_shape) - shape, 2, dtype=int)
        image_slices.append(tuple(slice(q[i], _shape[i]-q[i]-r[i]) for i in range(len(q))))
    return tuple(image_slices)
