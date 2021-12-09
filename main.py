import numpy as np
from core import *
import gpr_functions as gf
import gpflow as gpf
import matplotlib.pyplot as plt
import simulated_data as sd
import plotting

x = atleast_column(np.linspace(0, 10, 100))
fx = atleast_column(sd.smooth_data(100))

m = gf.simple_rbf_regression(x, fx)

fig, ax = plt.subplots()

plotting.model_plot(ax, m, (0, 10))
