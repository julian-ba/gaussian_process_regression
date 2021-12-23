from core import *
from skimage import io
import gaussian_process_functions as gpf

fname_in = "E:/test_funcMaps_fig3_LFwR_3waymap/save_fish17_fig3_LFwR_3waymap.mat_anat.tif"
fname_out = "C:/Users/jbaier/PycharmProjects/gaussian_process_regression/out.tif"

x, fx = sparsify(image, 0.01)
fx = fx.astype(np.dtype(float))

m = gpf.rbf_regression(x + 0.5, fx, lengthscales=5)

mean = m.predict_f(coord_list(
    np.arange(image_bounds[0]).astype(np.dtype(float)), np.arange(image_bounds[1]).astype(np.dtype(float)), np.arange(image_bounds[2]).astype(np.dtype(float))
))[0].numpy()

mean = mean.reshape(image_bounds[0], image_bounds[1], image_bounds[2]) * 100

io.imsave(fname_out, mean[0])
