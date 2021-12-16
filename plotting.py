# Plotting functions for GPR such as plotting distribution parameters from GPR over 1-d and 2-d data

import numpy as np
from core import *


def data_plot(ax, x, fx):
    ax.scatter(x, fx, c="k", marker="x")


def model_plot(ax, model, lower, upper):
    plot_points = atleast_column(np.linspace(lower, upper, 1000))
    mean_at_plot_points, var_at_plot_points = model.predict_f(plot_points)
    ax.plot(plot_points, mean_at_plot_points, "C0", lw=2)
    ax.fill_between(
        plot_points[:, 0],
        mean_at_plot_points[:, 0] - 1.96 * np.sqrt(var_at_plot_points[:, 0]),
        mean_at_plot_points[:, 0] + 1.96 * np.sqrt(var_at_plot_points[:, 0]),
        color="C0",
        alpha=0.2
    )


def sample_f_plot_1d(ax, model, lower, upper, n=10):
    plot_points = atleast_column(np.linspace(lower, upper, 1000))
    sampled_functions = model.predict_f_samples(plot_points, n)
    for sampled_function in sampled_functions:
        ax.plot(plot_points, sampled_function, color="C0", linewidth=0.5)


def sample_f_plot_2d(ax, model, lower, upper):
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], 1000), np.linspace(lower[1], upper[1], 1000))
    sampled_function = model.predict_f_samples(xx)
