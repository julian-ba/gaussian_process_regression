# Core plotting functions such as plotting a PDF over 1-d and 2-d data,

import numpy as np
import matplotlib.pyplot as plt
from core import *


def pos_plot(data_vector):
    if data_vector.dim == 1:
        if data_vector.binarized:
            plt.plot(np.arange(len(data_vector)), data_vector, "k.")
            plt.show()

    if data_vector.dim == 2:
        if data_vector.binarized:
            plt.plot(data_vector.x, data_vector.y, "k.")
            plt.show()
