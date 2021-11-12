# Core plotting functions such as plotting a PDF over 1-d and 2-d data,

import matplotlib.pyplot as plt
from core import *


def pos_plot(data_vector, save=False, path=None):
    if data_vector.binarized:
        if data_vector.dim == 1:
            plt.plot(np.arange(len(data_vector)), data_vector, "k.")
        elif data_vector.dim == 2:
            plt.plot(data_vector.x, data_vector.y, "k.")
        else:
            raise TypeError("Plotting for data of this dimension has not been implemented.")
    else:
        raise TypeError("Plotting for unbinarized data has not been implemented.")

    if save:
        if path is not None:
            fname = "{}/{}d-figure.png".format(path, data_vector.dim)
        else:
            fname = "{}d-figure.png".format(data_vector.dim)
        plt.savefig(fname)

    plt.show()
