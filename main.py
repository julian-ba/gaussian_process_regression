from testing import full_energy_distance, optimized_cumulative_energy_distance
import numpy as np

distributions = [np.zeros((100, 100)) for i in range(5)]
for distribution in distributions:
    for n in range(100):
        distribution[np.random.randint(0, 99), np.random.randint(0, 99)] += 1
validations = [distribution.copy() for distribution in distributions]

for validation in validations:
    for n in range(100):
        validation[np.random.randint(0, 99), np.random.randint(0, 99)] += 1

print("Full energy distance:")
print(full_energy_distance(distributions, validations))

print("Optimized energy distance:")
print(optimized_cumulative_energy_distance(distributions, validations))