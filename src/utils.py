from numpy.linalg import solve
from numpy import eye, empty


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
    LHS = gram_matrix + reg_param * gram_matrix.shape[-1] * np.eye(gram_matrix.shape[-1])
    return np.linalg.solve(LHS, outputs)


def generate_gram_matrix(input_features, kernel):
    """
    Create the gram matrix of inputs using a specific kernel
    """
    n_inputs = input_features.shape[0]
    G = empty((n_inputs, n_inputs))
    for i in range(n_inputs):
        for j in range(n_inputs):
            G[i, j] = kernel(input_features[i], input_features[j])
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
