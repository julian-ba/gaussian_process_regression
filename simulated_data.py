# Defines functions which create simulated data for testing purposes.
import numpy as np
import scipy.stats as stats
import numbers


def smooth_data(n, loc=0., scale=1., dim=None):
    # Returns n sample points from a multivariate Gaussian distribution distributed according to loc and scale.
    # The distributions in each dimension are independent.
    if dim is None:
        if isinstance(loc, numbers.Number):
            if isinstance(scale, numbers.Number):
                dim = 1
            else:
                try:
                    dim = len(scale)
                    loc = [loc] * dim
                except TypeError:
                    raise TypeError("Either dim, loc, or scale must be given.")
        else:
            try:
                dim = len(loc)
                if isinstance(scale, numbers.Number):
                    scale = [scale] * dim
            except TypeError:
                raise TypeError("Either dim, loc, or scale must be given.")

    if dim == 1:
        return stats.norm.rvs(size=n, loc=loc, scale=scale)

    elif dim > 1:
        if isinstance(loc, numbers.Number):
            if isinstance(scale, numbers.Number):
                return stats.norm.rvs(size=(n, dim), loc=loc, scale=scale)
            else:
                loc = [loc] * dim
        else:
            if isinstance(scale, numbers.Number):
                scale = [scale] * dim

        array = np.empty((n, dim))
        for i in range(dim):
            array[:, i] = stats.norm.rvs(size=n, loc=loc[i], scale=scale[i])

        return array


def jagged_data(n, loc=0., scale=1, dim=None):
    # Generates n points uniformly distributed over a hyper-cuboid. loc is the lower bound of the distribution (resp.
    # dimension), loc+scale is the upper bound.
    if dim is None:
        if isinstance(loc, numbers.Number):
            if isinstance(scale, numbers.Number):
                dim = 1
            else:
                try:
                    dim = len(scale)
                    loc = [loc] * dim
                except TypeError:
                    raise TypeError("Either dim, loc, or scale must be given.")
        else:
            try:
                dim = len(loc)
                if isinstance(scale, numbers.Number):
                    scale = [scale] * dim
            except TypeError:
                raise TypeError("Either dim, loc, or scale must be given.")

    if dim == 1:
        return stats.uniform.rvs(size=n, loc=loc, scale=scale)

    elif dim > 1:
        if isinstance(loc, numbers.Number):
            if isinstance(scale, numbers.Number):
                return stats.uniform.rvs(size=(n, dim), loc=loc, scale=scale)
            else:
                loc = [loc] * dim
        else:
            if isinstance(scale, numbers.Number):
                scale = [scale] * dim

        array = np.empty((n, dim))
        for i in range(dim):
            array[:, i] = stats.uniform.rvs(size=n, loc=loc[i], scale=scale[i])

        return array
