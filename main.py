from testing import cross_val_run
from image_processing import fish_fnames, import_tif_file

print(cross_val_run(import_tif_file(*fish_fnames, datatype=float), 10))
