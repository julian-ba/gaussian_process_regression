from testing import cross_val_run
import numpy as np
from image_processing import fish_fnames, import_tif_file
import matplotlib.pyplot as plt

y = cross_val_run(import_tif_file(*fish_fnames, datatype=float), 20)

means = y.mean(axis=1)

y = y/means

variance = y.var(axis=1)

x = np.concatenate((np.zeros(20), np.ones(20)))
y = y.flatten()

plt.scatter(x, y)
plt.errorbar(np.array([0, 1]), means, yerr=variance)
plt.savefig("error.png")
