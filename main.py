import GPy
import numpy as np

import simulated_data as sd
from matplotlib import pyplot as plt
from core import *

X = np.column_stack((np.linspace(0, 1, 100), np.linspace(0, 1, 100)))
Y = np.column_stack((X, sd.smooth_data(100, loc=2, dim=1)))

RBF_kernel = GPy.kern.RBF(input_dim=2)
plt.show()
model = GPy.models.gp_regression.GPRegression(X, Y, RBF_kernel)

