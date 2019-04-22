import numpy as np


def bernoulli(p, size):
    return np.random.binomial(1, p, size)
