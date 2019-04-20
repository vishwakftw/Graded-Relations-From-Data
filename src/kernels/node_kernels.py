from numpy.linalg import norm
from math import exp


def gaussian_kernel(x, y, width=1.0):
    """
    Gaussian (RBF) kernel: exp(-width ||x - y||^2)
    where ||.|| denotes the Euclidean distance
    """
    distance = norm(x - y)
    return exp(-width * distance ** 2)
