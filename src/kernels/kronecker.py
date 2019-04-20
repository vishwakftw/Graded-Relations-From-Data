from node_kernels import gaussian_kernel


def kronecker_product_pairwise_kernel(edge1, edge2, **kwargs):
    """
    Computes the Kronecker product pairwise kernel between two edges.
    Uses the Gaussian (RBF) kernel internally for nodes.
    """
    x1, y1 = edge1
    x2, y2 = edge2
    return gaussian_kernel(x1, x2, **kwargs) * gaussian_kernel(y1, y2, **kwargs)


def reciprocal_kronecker_product_pairwise_kernel(edge1, edge2, **kwargs):
    """
    Computes the Reciprocal Kronecker product pairwise kernel between two edges.
    Uses the Gaussian (RBF) kernel internally for nodes.
    """
    x1, y1 = edge1
    x2, y2 = edge2
    part1 = gaussian_kernel(x1, x2, **kwargs) * gaussian_kernel(y1, y2, **kwargs)
    part2 = gaussian_kernel(x1, y2, **kwargs) * gaussian_kernel(x2, y1, **kwargs)
    return 2 * (part1 - part2)


def symmetric_kronecker_product_pairwise_kernel(edge1, edge2, **kwargs):
    """
    Computes the Reciprocal Kronecker product pairwise kernel between two edges.
    Uses the Gaussian (RBF) kernel internally for nodes.
    """
    x1, y1 = edge1
    x2, y2 = edge2
    part1 = gaussian_kernel(x1, x2, **kwargs) * gaussian_kernel(y1, y2, **kwargs)
    part2 = gaussian_kernel(x1, y2, **kwargs) * gaussian_kernel(x2, y1, **kwargs)
    return 2 * (part1 + part2)


def metric_learning_pairwise_kernel(edge1, edge2, **kwargs):
    """
    Computes the Metric Learning pairwise kernel between two edges.
    Uses the Gaussian (RBF) kernel internally for nodes.
    """
    x1, y1 = edge1
    x2, y2 = edge2
    corr_x1_y1 = gaussian_kernel(x1, y1, **kwargs)
    corr_x2_y2 = gaussian_kernel(x2, y2, **kwargs)
    corr_x1_y2 = gaussian_kernel(x1, y2, **kwargs)
    corr_x2_y1 = gaussian_kernel(x2, y1, **kwargs)
    return (corr_x1_y1 + corr_x2_y2 - corr_x1_y2 - corr_x2_y1) ** 2
