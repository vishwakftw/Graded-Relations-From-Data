import grd.kernels as kernels

from .distributions import bernoulli
from numpy.linalg import solve
from numpy import eye, empty, exp, heaviside, reciprocal


def _sigmoid(x):
    """
    Sigmoid function
    """
    return reciprocal(1 + exp(-x))


def solver(gram_matrix, outputs, reg_param):
    """
    Obtain the dual weights by solving the system of linear equations:
    (G + n lambda I) a = Y
    where `G` is the gram matrix of inputs of size n x n
          `lambda` is the regularization constant
          `I` is the identity matrix of size of n x n
          `a` are the dual weights
          `Y` is the vector of outputs of length n
    """
    LHS = gram_matrix + reg_param * gram_matrix.shape[-1] * eye(gram_matrix.shape[-1])
    return solve(LHS, outputs)


def generate_gram_matrix(input_features, kernel, **kwargs):
    """
    Create the gram matrix of inputs using a specific kernel
    """
    n_inputs = input_features.shape[0]
    G = empty((n_inputs, n_inputs))
    for i in range(n_inputs):
        for j in range(n_inputs):
            G[i, j] = kernel(input_features[i], input_features[j], **kwargs)
    return G


def compute_similarity(A, B, t, t_dash, u, v):
    """
    Given two binary vectors A and B, computes the similarity metric parameterized
    by t, t_dash, u, v

                  (t * |sim_diff(A, B)| + u * |intersection(A, B)| + v * |intersection(Ac, Bc)|)
   S_{A, B} =  -----------------------------------------------------------------------------------
               (t_dash * |sim_diff(A, B)| + u * |intersection(A, B)| + v * |intersection(Ac, Bc)|)
    """
    A_bool = A.astype(bool)
    B_bool = B.astype(bool)
    sim_diff = sum(A_bool ^ B_bool)
    intersect = sum(A_bool & B_bool)
    union_comp = sum(~(A_bool | B_bool))
    common_term = u * intersect + v * union_comp
    return (t * sim_diff + common_term) / (t_dash * sim_diff + common_term)


def heaviside_similarity(edge):
    """
    Function to compute the heaviside similarity between the feature
    representation of the vertices between an edge.
    """
    f1, f2 = edge
    assert len(f1) == len(f2), "Vector length mismatch"
    similarity = 0.0
    for e1, e2 in zip(f1, f2):
        similarity += heaviside(e1, e2)
    similarity /= len(f1)
    return similarity


def get_predictor(h, sigma, b):
    """
    TODO
    """
    def q(x):
        h_x = h(x)
        if h_x < -b:
            return 0
        elif h_x >= -b and h_x <= b:
            return sigma(h_x)
        else:
            return 1
    return q


def map_kernel(kernel_name):
    """
    Function to map a kernel function to the callable.
    """
    if kernel_name.find('kronecker') != -1:
        return getattr(kernels.kronecker, kernel_name + '_product_pairwise_kernel')
    elif kernel_name == 'cartesian':
        return kernels.cartesian.cartesian_pairwise_kernel
    elif kernel_name == 'mlpk':
        return kernels.kronecker.metric_learning_pairwise_kernel
    else:
        raise NotImplementedError('Kernel {} not implemented'.format(kernel_name))


def map_sigma(sigma_name):
    """
    Function to map a sigma function to the callable.
    """
    if sigma_name == 'sigmoid':
        return _sigmoid
    else:
        raise NotImplementedError('sigma function {} not implemented'.format(sigma_name))


def map_similarity_vector(similarity_vector_name):
    """
    Function to get the similarity vector parameterization given the name.
    """
    if similarity_vector_name == 'jaccard':
        return [0, 1, 1, 0]
    elif similarity_vector_name == 'sokal':
        return [0, 1, 2, 2]
    elif similarity_vector_name == 'tp':
        return [1, 2, 1, 1]
    else:
        raise ValueError('Similarity vector form for {} not available'
                         .format(similarity_vector_name))


def map_dist_func(dist_func_name):
    """
    Function to map a distribution generation function to the callable.
    """
    if dist_func_name == 'bernoulli':
        return bernoulli
    else:
        raise NotImplementedError('dist_func {} not implemented'.format(dist_func_name))
