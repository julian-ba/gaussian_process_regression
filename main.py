import numpy as np
import os
import matplotlib.pyplot as plt
import simulated_data as sd
import plotting
import timing
from core import *

x = np.linspace(0, 10, 11)
y = np.linspace(0, 10, 11)

xy = coord_list(x, y)

fx = sd.jagged_data(121)

m = rbf_regression(xy, fx, lengthscales=0.5)

fig = plt.figure()
ax = plt.axes(projection="3d")

plotting.model_plot_2d(ax, m, 0, 10)
plt.show()
