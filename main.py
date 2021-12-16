import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import timing
from core import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tested_n, result_array = timing.run_1d_test(10, 1, 10001, 100, return_seconds=True)

plt.plot(tested_n, result_array)
plt.ylabel("Mean time taken in seconds")
plt.xlabel("n")
plt.savefig("Without GPU.png")
