from testing import cross_val_run
import numpy as np
from image_processing import fish_fnames, import_tif_file, export_tif_file
from testing import cross_val_run
import matplotlib.pyplot as plt

ffish = import_tif_file(*fish_fnames)

print(cross_val_run(ffish, 5, output_image=True))
