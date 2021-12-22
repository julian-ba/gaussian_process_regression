from core import *
from skimage import io
import gaussian_process_functions as gpf

fname_in = ""
fname_out = ""

image = io.imread(fname_in)
image_bounds = image.shape
x, fx = sparsify(image, 1)

m = gpf.rbf_regression(x + 0.5, fx)

mean = m.predict_f(coord_list(np.arange(image_bounds[0]) + 0.5, np.arange(image_bounds[1]) + 0.5))[0].numpy()
mean = mean.reshape(image_bounds[0], image_bounds[1])

io.imsave(fname_out, mean)
