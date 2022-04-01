from matplotlib import pyplot as plt
from user_values import FILE_NAMES
from testing import optimize_gpr, optimize_kde
import numpy as np


lengthscales_range = np.geomspace(1, 20, 6)

optimization_gpr = optimize_gpr(FILE_NAMES, lengthscales_range, iterations=1, output=True, distribution_representation="cdf")
plt.plot(optimization_gpr["lengthscales"], optimization_gpr["score"])
plt.scatter(optimization_gpr["lengthscales"], optimization_gpr["score"])
plt.xlabel("Lengthscales")
plt.ylabel("Cumulative per-pixel energy distance")
plt.savefig("figures/optimization_gpr.pdf")
plt.close()



sigma_range = np.geomspace(1, 80, 12)

optimization_kde = optimize_kde(FILE_NAMES, sigma_range, iterations=5)
plt.plot(optimization_kde["sigma"], optimization_kde["score"])
plt.scatter(optimization_kde["sigma"], optimization_kde["score"])
plt.xlabel("Sigma")
plt.ylabel("Cumulative per-pixel energy distance")
plt.savefig("figures/optimization_kde.pdf")
plt.close()
