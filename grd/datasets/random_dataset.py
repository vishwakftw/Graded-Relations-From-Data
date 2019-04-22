import numpy as np
import grd.utils as utils

from ..distributions import bernoulli


def _generate_subsection(X, y, indices):
    return X[indices], y[indices]


class RandomDataset(object):

    def __init__(self, size, num_nodes, dims, similarity_vector, p=0.5, dist_func=bernoulli, share_nodes=True):
        assert size > 0 and num_nodes > 0 and dims > 0, \
               "Invalid size, node count, or dims specified"
        self.dims = dims
        self.num_nodes = num_nodes
        self.size = size
        self.share_nodes = share_nodes
        if share_nodes:
            self.nodes = dist_func(p, [num_nodes, dims])
            self.similarity_vector = similarity_vector
            self.X, self.y = self.generate_data(self.nodes, size, similarity_vector)
        else:
            self.train_nodes = dist_func(p, [num_nodes, dims])
            self.val_nodes = dist_func(p, [num_nodes, dims])
            self.test_nodes = dist_func(p, [num_nodes, dims])
            self.train_X, self.train_y = self.generate_data(self.train_nodes, size, similarity_vector)
            self.val_X, self.val_y = self.generate_data(self.val_nodes, size, similarity_vector)
            self.test_X, self.test_y = self.generate_data(self.test_nodes, size, similarity_vector)

    def generate_data(self, nodes, size, similarity_vector):
        pairs = np.array([(x, y) for x in nodes for y in nodes if np.any(x != y)])
        edges = np.random.permutation(pairs)[:size]
        labels = np.array([utils.compute_similarity(*e, *similarity_vector) for e in edges])
        return edges, labels

    def add_noise(self, fraction):
        assert fraction >= 0.0 and fraction <= 1.0, "Invalid fraction specified"
        for i in range(self.size):
            for j in range(2):
                indices = np.random.choice(self.dims, int(self.dims * fraction))
                if self.share_nodes:
                    self.X[i, j, indices] = 1 - self.X[i, j, indices]
                else:
                    self.train_X[i, j, indices] = 1 - self.train_X[i, j, indices]
                    self.val_X[i, j, indices] = 1 - self.val_X[i, j, indices]
                    self.test_X[i, j, indices] = 1 - self.test_X[i, j, indices]

    def train_val_test_split(self, train_fraction=None, val_fraction=None,
                             train_size=None, val_size=None):
        assert (train_fraction is not None and val_fraction is not None) or \
               (train_size is not None and val_size is not None), "Split improperly specified"
        if train_fraction is None:
            assert train_size >= 0 and val_size >= 0 and \
                   train_size + val_size <= self.size, "Invalid size"
            train_split_index = train_size
            val_split_index = train_size + val_size
        else:
            assert train_fraction >= 0.0 and val_fraction >= 0.0 and \
                   train_fraction + val_fraction <= 1.0, "Invalid fractions"
            train_split_index = int(self.size * train_fraction)
            val_split_index = int(self.size * (train_fraction + val_fraction))

        permutation = np.random.permutation(np.arange(self.size))
        self.train_indices, self.val_indices, self.test_indices = \
            np.split(permutation, [train_split_index, val_split_index])
        self.train_X, self.train_y = _generate_subsection(self.X, self.y, self.train_indices)
        self.val_X, self.val_y = _generate_subsection(self.X, self.y, self.val_indices)
        self.test_X, self.test_y = _generate_subsection(self.X, self.y, self.test_indices)

    def cv_x(self):
        return np.vstack((self.train_X, self.val_X))

    def cv_y(self):
        return np.hstack((self.train_y, self.val_y))

    def cv_size(self):
        return self.train_X.shape[0] + self.val_X.shape[0]

    def cv_indices(self):
        return np.arange(self.train_X.shape[0], self.train_X.shape[0] + self.val_X.shape[0])
