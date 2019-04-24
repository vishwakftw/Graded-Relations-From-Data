from .node_kernels import gaussian_kernel


def cartesian_pairwise_kernel(edge1, edge2, **kwargs):
    """
    Computes the Cartesian pairwise kernel between two edges.
    Uses the Gaussian (RBF) kernel internally for nodes.
    """
    x1, y1 = edge1
    x2, y2 = edge2
    value = gaussian_kernel(x2, y2, **kwargs) + gaussian_kernel(x1, y1, **kwargs)
    return value
