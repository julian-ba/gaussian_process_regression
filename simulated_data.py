# Defines functions which create simulated data for testing purposes.
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions

import numpy as np
import numbers


def smooth_data(n, loc=0., scale=1., return_variable=False):
    if return_variable:
        return tf.Variable(tfd.Normal(loc, scale).sample(n))
    else:
        return tfd.Normal(loc, scale).sample(n)
