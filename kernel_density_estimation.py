from core import *


def format_function(func):
    def function(x):
        out = np.empty(len(x))
        for xi in range(len(x)):
            out[xi] = func(xi)
        return exactly_2d(out)
    return function


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
    return lambda x: gaussian_with_mean_and_var.pdf


def radial_gaussian(std):
    return lambda x: to_radial_function(gaussian(0, std))(x)


def bandwidth_rule_of_thumb(n, std):
    if n == 0:
        return 0.
    else:
        return np.power(4 * std**5 / (3*n), 1/5)


def linear(a, b):
    return lambda x: (x - a)/b


def kernel_density_estimation(x, kernel, eps=1e-9, step=None):
    from scipy.signal import convolve
    if step is None:
        epsi = 0
        while kernel(epsi) > eps:
            epsi += 1
        epsilon = np.full(x.ndim, epsi)
    else:
        step_min = np.min(np.array(step))
        epsi = 0
        while kernel(epsi) > eps:
            epsi += step_min
        try:
            assert epsi > 0
            epsilon = np.ceil(epsi/np.array(step)).astype(int)
        except AssertionError:
            epsilon = 0
    slices = [slice(-epsi, epsi+1) for epsi in epsilon]
    grid = Grid(slices, step)
    evaluate = grid.get_list()
    convolution_matrix = grid.to_array(kernel(evaluate))
    return convolve(x, convolution_matrix, "same")


def gaussian_kernel_density_estimation(x, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x, sigma)
