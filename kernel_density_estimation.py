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


def kernel_density_estimator(x, h, kernel):
    # TODO: Improve efficiency for large sets of x
    n = len(x)

    def estimator(_x):
        output_n = len(_x)
        if n == 0:
            return np.zeros((output_n, 1))

        else:
            raw_output = np.empty((output_n, n))
            with np.nditer(raw_output, flags=["multi_index"], op_flags=["writeonly"]) as it:
                for i in it:
                    j, k = it.multi_index
                    i[...] = kernel(linear(x[k], h)(_x[j]))

            return np.sum(raw_output, 1, keepdims=True)

    return estimator
