# Defines functions which create simulated data for testing purposes.
from core import *
import numpy as np
import scipy.stats as stats


def smooth_data(n, loc, scale, ndim=3):
    # Returns n sample points as a SpatialPointVector from a 3-variate Gaussian distribution distributed according to.
    # The distributions in each dimension are independent.
    if ndim == 1:
        return SpatialPointVector(stats.norm.rvs(size=n, loc=loc, scale=scale), dim=ndim)

    elif ndim == 2:
        array = np.column_stack((
            stats.norm.rvs(size=n, loc=loc[0], scale=scale[0]),
            stats.norm.rvs(size=n, loc=loc[1], scale=scale[1]))
        )
        return SpatialPointVector(array)

    elif ndim == 3:
        array = np.column_stack((
            stats.norm.rvs(size=n, loc=loc[0], scale=scale[0]),
            stats.norm.rvs(size=n, loc=loc[1], scale=scale[1]),
            stats.norm.rvs(size=n, loc=loc[2], scale=scale[2]))
        )
        return SpatialPointVector(array)
    else:
        raise TypeError("Dimensions other than 1, 2, or 3 are not implemented.")


def jagged_data(n, loc, scale, ndim=3):
    # Generates n points uniformly distributed over a cuboid. loc is the lower bound of the distribution (resp.
    # dimension), loc+scale is the upper bound.
    if ndim == 1:
        return SpatialPointVector(stats.uniform.rvs(size=n, loc=loc, scale=scale), dim=ndim)

    elif ndim == 2:
        array = np.column_stack((
            stats.uniform.rvs(size=n, loc=loc[0], scale=scale[0]),
            stats.uniform.rvs(size=n, loc=loc[1], scale=scale[1]))
        )
        return SpatialPointVector(array)

    elif ndim == 3:
        array = np.column_stack((
            stats.uniform.rvs(size=n, loc=loc[0], scale=scale[0]),
            stats.uniform.rvs(size=n, loc=loc[1], scale=scale[1]),
            stats.uniform.rvs(size=n, loc=loc[2], scale=scale[2]))
        )
        return SpatialPointVector(array)

    else:
        raise TypeError("Dimensions other than 1, 2, or 3 are not implemented.")
