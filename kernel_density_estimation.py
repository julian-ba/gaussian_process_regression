from core import *


def euclidean_norm(x):
    return np.sqrt(np.sum(np.square(x)))


def euclidean_distance(x, y):
    return euclidean_norm(x - y)


def to_radial_function(function):
    return lambda x: function(euclidean_norm(x))


def find_bounds_for_gaussian(sigma, epsilon=np.finfo(float).eps/2):
    # cf. Proposition 1.2 on p.1 in mathematical_foundations.pdf
    return sigma * np.sqrt(-2*np.log(np.sqrt(2 * np.pi) * epsilon/sigma))


def gaussian(mean, std):
    from scipy import stats
    gaussian_with_mean_and_var = stats.norm(loc=mean, scale=std)
    return gaussian_with_mean_and_var.pdf


def radial_gaussian(mean, std):
    return lambda x: to_radial_function(gaussian(0, std))(x - mean)


def bandwidth_rule_of_thumb(n, std):
    if n == 0:
        return 0.
    else:
        return np.power(4 * std**5 / (3*n), 1/5)


def linear(a, b):
    return lambda x: (x - a)/b


def naive_kernel_density_estimation(x, h, kernel, epsilon, step=None):
    # !!! Do not use: very slow
    from scipy.signal import convolve

    epsilon = np.broadcast_to(epsilon, (x.shape[1],))
    convolution_matrix = np.empty(epsilon)

    if step is not None:
        step = np.broadcast_to(step, (x.shape[1],))


def gaussian_kernel_density_estimation(x, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x, sigma)


def gaussian_kernel_density_estimator(sigma):
    return lambda x: gaussian_kernel_density_estimation(x, sigma)
