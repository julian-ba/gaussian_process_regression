import numpy as np

import image_processing
import testing
from skimage import img_as_float

fish = image_processing.import_tif_file(*image_processing.fish_fnames, datatype=np.dtype(float))
print(testing.cross_val_run(fish, 20))
