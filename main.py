import numpy as np
from core import *
import gpr_functions as gf
import gpflow as gpf
import matplotlib.pyplot as plt
import simulated_data as sd
import plotting

x = atleast_column(np.linspace(0, 10, 11))
fx = atleast_column(sd.smooth_data(11))

m = gf.simple_rbf_regression(x, fx, lengthscales=0.5)

fig, ax = plt.subplots(figsize=(16, 8))

plotting.data_plot(ax, x, fx)
plotting.model_plot(ax, m, (0, 10))
plotting.sample_f_plot(ax, m, (0, 10))

fig.savefig("plot_model.png", dpi=300)
