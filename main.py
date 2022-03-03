from testing import cross_val_run
from image_processing import fish_fnames, import_tif_file
from testing import cross_val_run
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from gaussian_process_regression import rbf_regression_over_large_array

print(cross_val_run(fish_fnames, 5, output_image=True))
