from testing import optimize_parameters
from image_processing import fish_fnames
import numpy as np
from kernel_density_estimation import gaussian_kernel_density_estimation
from matplotlib import pyplot as plt
N = 50

array = optimize_parameters(fish_fnames, gaussian_kernel_density_estimation, {"sigma": np.linspace(0.5, 100, 50)}, iterations=N)
plt.plot(array["sigma"], array["score"])
plt.scatter(array["sigma"], array["score"])
plt.xlabel("sigma")
plt.ylabel("Score over {} iterations".format(N))
plt.savefig("figures/parameter_optimization_kde.png")
