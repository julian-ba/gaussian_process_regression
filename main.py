from matplotlib import pyplot as plt

import core
from plotting import prediction_histograms
from testing import random_indices
import numpy as np
from gaussian_process_regression import maximum_likelihood_rbf_regression_over_large_array
from kernel_density_estimation import gaussian_kernel_density_estimation

from image_processing import FISH_FNAMES, import_tif_file, array_crops, shape_from_slice

ffish_crops = array_crops(*import_tif_file(*FISH_FNAMES))
fish_shapes = shape_from_slice(*ffish_crops[0][1:])
middle_z = [(i[0].stop - i[0].start)//2 for i in ffish_crops]
splits1, splits2 = random_indices(2, len(FISH_FNAMES))
agglomerated_fish = np.zeros(fish_shapes)
for k in splits1:
    agglomerated_fish += import_tif_file(FISH_FNAMES[k], key=middle_z[k])[ffish_crops[k][1:]]
agglomerated_fish /= len(splits1)
prediction = maximum_likelihood_rbf_regression_over_large_array(agglomerated_fish, threshold=0.05, lengthscales=4., step=core.GRID_STEP_2D, it_step=(200, 200))

agglomerated_fish = np.zeros(fish_shapes)
for k in splits2:
    agglomerated_fish += import_tif_file(FISH_FNAMES[k], key=middle_z[k])[ffish_crops[k][1:]]
agglomerated_fish /= len(splits2)

ffish_crops = array_crops(*import_tif_file(*FISH_FNAMES))
fish_shapes = shape_from_slice(*ffish_crops[0][1:])
middle_z = [(i[0].stop - i[0].start)//2 for i in ffish_crops]
splits1, splits2 = random_indices(2, len(FISH_FNAMES))
agglomerated_fish = np.zeros(fish_shapes)
for k in splits1:
    agglomerated_fish += import_tif_file(FISH_FNAMES[k], key=middle_z[k])[ffish_crops[k][1:]]
agglomerated_fish /= len(splits1)
prediction = gaussian_kernel_density_estimation(agglomerated_fish, 8.)

agglomerated_fish = np.zeros(fish_shapes)
for k in splits2:
    agglomerated_fish += import_tif_file(FISH_FNAMES[k], key=middle_z[k])[ffish_crops[k][1:]]
agglomerated_fish /= len(splits2)