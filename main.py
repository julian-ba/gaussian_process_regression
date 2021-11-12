import core
import simulated_data as sd
from core import *
import plot_functions as plot

plot.pos_plot(sd.smooth_data(100, [1, 1], [1, 1], 2), True)
