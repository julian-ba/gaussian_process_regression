# Plotting functions for GPR such as plotting distribution parameters from GPR over 1-d and 2-d data
import numpy as np
from core import *


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


def model_plot_2d(ax1, ax2, model, slices, step=None):
    grid = Grid(slices, step=step)
    plot_points = grid.get_list()
    xx = grid.get_array()[..., 0]
    yy = grid.get_array()[..., 1]

    sampled_function_mean, sampled_function_var = model.predict_f(plot_points)
    sampled_function_mean = grid.to_array(sampled_function_mean.numpy())
    sampled_function_var = grid.to_array(sampled_function_var.numpy())
    ax1.pcolormesh(xx, yy, sampled_function_mean)
    ax2.pcolormesh(xx, yy, sampled_function_var)


def sample_f_plot_1d(ax, model, lower, upper, n=10):
    plot_points = exactly_2d(np.linspace(lower, upper, 1000))
    sampled_functions = model.predict_f_samples(plot_points, n)
    for sampled_function in sampled_functions:
        ax.plot(plot_points, sampled_function, color="C0", linewidth=0.5)


def sample_f_plot_2d(ax, model, slices, step=None):
    grid = Grid(slices, step=step)
    plot_points = grid.get_list()
    xx = grid.get_array()[0]
    yy = grid.get_array()[1]

    sampled_func = model.predict_f_samples(plot_points)
    sampled_func = grid.to_array(sampled_func.numpy())
    ax.plot_surface(xx, yy, sampled_func)


def histogram_and_image_figure(prediction: np.ndarray, validation: np.ndarray, file_name: str):
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True, gridspec_kw={"width_ratios": [1, 3]})
    ((prediction_histogram, prediction_image), (validation_histogram, validation_image)) = axs
    if prediction.shape[0] < prediction.shape[0]:
        prediction_array = prediction.T
    else:
        prediction_array = prediction
    if validation.shape[0] < validation.shape[1]:
        validation_array = validation.T
    else:
        validation_array = validation
    vmin, vmax = (
    min(prediction_array.amin(), validation_array.amin()), max(prediction_array.amax(), validation_array.amax()))
    range_ = (vmin, vmax)
    print(range_)
    prediction_histogram.hist(prediction_array.ravel(), bins=40, range=range_, log=True)
    prediction_histogram.set_title("prediction array histogram"),
    validation_histogram.hist(validation_array.ravel(), bins=40, range=range_, log=True)
    validation_histogram.set_title("validation array histogram")
    prediction_image.imshow(prediction_array, vmin=vmin, vmax=vmax)
    validation_image.imshow(validation_array, vmin=vmin, vmax=vmax)
    fig.savefig(file_name)
