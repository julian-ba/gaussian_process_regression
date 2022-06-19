import numpy as np
from image_processing import import_tif_files
from core import array_crops
from user_values import MEMORY_STEP_3D
from gaussian_process_regression import rbf_regression_over_large_array
from testing import optimized_cumulative_energy_distance, generate_gaussian_samples
from image_processing import export_tif_file
import tensorflow as tf

fish_images = import_tif_files(dtype=np.dtype(np.float16))
array_crop = array_crops(*fish_images, shape=[10, 900, 500])
images = [fish_images[i][array_crop[i]] for i in range(len(fish_images))]

split = len(images)//2

prediction_mean, prediction_var = rbf_regression_over_large_array(*images[:-split], float_type=np.float16, lengthscales=12, it_step=MEMORY_STEP_3D, include_zeros=True, method="fast", do_optimization=False)
export_tif_file("mean", prediction_mean)
points = generate_gaussian_samples(prediction_mean, np.sqrt(prediction_var), num=5)
export_tif_file("sample", prediction_mean)
print(optimized_cumulative_energy_distance(points, images[split:]))


