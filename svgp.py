import gpflow
import numpy as np
from user_values import MEMORY_STEP_2D


def rbf_sparse_model(x, fx, num_inducing=200, variance=1., lengthscales=1.):
    indices = np.random.randint(0, len(x), dtype=np.int64)
    gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(variance, lengthscales), likelihood=gpflow.likelihoods.Gaussian(), inducing_variable=x[indices])


def rbf_sparse_regression():
