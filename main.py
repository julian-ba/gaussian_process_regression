import numpy as np
from core import *

print(coord_list(np.linspace(0, 1, 11), np.linspace(0, 10, 11), np.linspace(0, 100, 11)))

space = np.zeros((11, 11, 11))
space[6, 7, 8] = 1
print(np.argwhere(space == 1))
