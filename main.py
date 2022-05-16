from testing import full_energy_distance, optimized_cumulative_energy_distance
import numpy as np
from image_processing import import_tif_files
from core import array_crops
from user_values import MEMORY_STEP_3D
from gaussian_process_regression import rbf_regression_over_large_array
from testing import full_energy_distance, optimized_cumulative_energy_distance, generate_gaussian_points_with_weights


fish_images = import_tif_files(dtype=np.dtype(np.float16))
array_crop = array_crops(*fish_images, shape=[10, 900, 500])
images = [fish_images[i][array_crop[i]] for i in range(len(fish_images))]

split = len(images)//2

prediction_mean, prediction_var = rbf_regression_over_large_array(*images[:-split], lengthscales=12, it_step=MEMORY_STEP_3D, include_zeros=False, method="slow")
points, weights = generate_gaussian_points_with_weights(prediction_mean, prediction_var, "cdf")

print(full_energy_distance(*weights[:], images[split:], points))
