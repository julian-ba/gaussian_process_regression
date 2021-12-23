# Plotting functions for GPR such as plotting distribution parameters from GPR over 1-d and 2-d data

import numpy as np
from core import *
import matplotlib.pyplot as plt


def data_plot_1d(ax, x, fx):
    ax.scatter(x, fx, c="k", marker="x")


def model_plot_1d(ax, model, lower, upper):
    plot_points = exactly_2d(np.linspace(lower, upper, 1000))
    mean_at_plot_points, var_at_plot_points = model.predict_f(plot_points)
    ax.plot(plot_points, mean_at_plot_points, "C0", lw=2)
    ax.fill_between(
        plot_points[:, 0],
        mean_at_plot_points[:, 0] - 1.96 * np.sqrt(var_at_plot_points[:, 0]),
        mean_at_plot_points[:, 0] + 1.96 * np.sqrt(var_at_plot_points[:, 0]),
        color="C0",
        alpha=0.2
    )


def model_plot_2d(ax1, ax2, model, lower, upper):
    if not hasattr(lower, "__getitem__"):
        lower = [lower, lower]

    if not hasattr(upper, "__getitem__"):
        upper = [upper, upper]

    x = np.linspace(lower[0], upper[0], 100)
    y = np.linspace(lower[1], upper[1], 100)

    plot_points, (xx, yy) = coord_list_and_meshgrid(x+0.5, y+0.5)

    sampled_function_mean, sampled_function_var = model.predict_f(plot_points)
    sampled_function_mean = sampled_function_mean.numpy().reshape(len(x), len(y))
    sampled_function_var = sampled_function_var.numpy().reshape(len(x), len(y))
    ax1.pcolormesh(xx, yy, sampled_function_mean)
    ax2.pcolormesh(xx, yy, sampled_function_var)


def sample_f_plot_1d(ax, model, lower, upper, n=10):
    plot_points = exactly_2d(np.linspace(lower, upper, 1000))
    sampled_functions = model.predict_f_samples(plot_points, n)
    for sampled_function in sampled_functions:
        ax.plot(plot_points, sampled_function, color="C0", linewidth=0.5)


def sample_f_plot_2d(ax, model, lower, upper):
    if not hasattr(lower, "__getitem__"):
        lower = [lower, lower]

    if not hasattr(upper, "__getitem__"):
        upper = [upper, upper]

    x = np.linspace(lower[0], upper[0], 100)
    y = np.linspace(lower[1], upper[1], 100)

    plot_points, (xx, yy) = coord_list_and_meshgrid(x, y)

    sampled_func = model.predict_f_samples(plot_points)
    sampled_func = sampled_func.numpy().reshape(x, y)
    ax.plot_surface(xx, yy, sampled_func)
