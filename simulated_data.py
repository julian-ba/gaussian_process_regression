# Defines functions which create simulated data for testing purposes.
import numpy as np
import scipy.stats as stats


def smooth_data(n, loc=0., scale=1., dim=None):
    # Returns n sample points from a multivariate Gaussian distribution distributed according to loc and scale.
    # The distributions in each dimension are independent.
    loc, scale = np.broadcast_arrays(loc, scale)

    array = np.empty((n, loc.size))
    for i in range(loc.size):
        array[:, i] = stats.norm.rvs(size=n, loc=loc[i], scale=scale[i])

    return array


def jagged_data(n, loc=0., scale=1, dim=None):
    # Generates n points uniformly distributed over a hyper-cuboid. loc is the lower bound of the distribution (resp.
    # dimension), loc+scale is the upper bound.
    loc, scale = np.broadcast_arrays(loc, scale)

    array = np.empty((n, loc.size))
    for i in range(loc.size):
        array[:, i] = stats.uniform.rvs(size=n, loc=loc[i], scale=scale[i])

    return array
