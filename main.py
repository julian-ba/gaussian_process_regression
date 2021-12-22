import numpy as np
from core import *

space = np.arange(1000).reshape((10, 10, 10))

print(space[5][2][1])

print(coord_list(100*np.arange(10), 10*np.arange(10), np.arange(10))[521])
