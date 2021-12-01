import GPy as gpy
import numpy as np
import numpy.random

import simulated_data as sd
from matplotlib import pyplot as plt
from core import *

model = gpy.examples.regression.sparse_GP_regression_1D(10, 2, 0, False, True)
arr = np.array([[2]])
print(arr)
mean, var = model.predict(arr)

print(mean)
print(var)
