import GPy as gpy
from numpy import concatenate
from core import *


def predict_per_pixel(posterior_points, prior_points, values, kernel, noise_var=1e-4):
    model = gpy.models.gp_regression.GPRegression(
        atleast_column(prior_points), atleast_column(values), kernel=kernel, noise_var=noise_var
    )
    return model.predict(posterior_points)
