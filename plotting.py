# Plotting functions for GPR such as plotting distribution parameters from GPR over 1-d and 2-d data

import matplotlib.pyplot as plt
import numpy as np

from core import *


def data_plot(ax, x, fx):
    out = ax.scatter(x, fx, c="k", marker="x")
    return out


def model_plot(ax, model, interval):
    plot_points = atleast_column(np.linspace(interval[0], interval[1], 100))
    mean_at_plot_points, var_at_plot_points = model.predict_f(plot_points)
    out = ax.plot(plot_points, mean_at_plot_points, "C0", lw=2)
    print(out)
    out = ax.fill_between(
        plot_points[:, 0],
        mean_at_plot_points[:, 0] - 1.96 * np.sqrt(var_at_plot_points[:, 0]),
        mean_at_plot_points[:, 0] + 1.96 * np.sqrt(var_at_plot_points[:, 0]),
        color="C0",
        alpha=0.2
    )
    return out
