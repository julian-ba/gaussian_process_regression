from matplotlib import pyplot as plt
from image_processing import FISH_FNAMES
from testing import optimize_gpr
import numpy as np

lengthscales_range = np.geomspace(5, 80, 10)

optimization_gpr = optimize_gpr(FISH_FNAMES, lengthscales_range, iterations=5, output=True)
plt.plot(optimization_gpr["lengthscales"], optimization_gpr["score"])
plt.scatter(optimization_gpr["lengthscales"], optimization_gpr["score"])
plt.xlabel("Length-scale")
plt.ylabel("Cumulative per-pixel energy distance")
plt.savefig("figures/optimization_gpr.pdf")
