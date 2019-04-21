import numpy as np
import utils

from ..distributions import bernoulli


class RandomDataset(object):

    def __init__(self, size, num_nodes, dims, similarity_vector, p=0.5, dist_func=bernoulli):
        assert size > 0 and num_nodes > 0 and dims > 0, "Invalid size, node count, or dims specified"
        self.dims = dims
        self.num_nodes = num_nodes
        self.size = size
        self.nodes = dist_func(p, [num_nodes, dims])
        self.similarity_vector = similarity_vector
        self.X, self.y = self.generate_data(self.nodes, size, similarity_vector)

    def generate_data(self, nodes, size, similarity_vector):
        pairs = np.array([(x, y) for x in nodes for y in nodes if x != y])
        edges = np.random.permutation(pairs)[:size]
        labels = np.array([utils.similarity(e, similarity_vector) for e in edges])
        return edges, labels

    def permute_data(self):
        permutation = np.random.permutation(np.arange(self.size))
        self.X = self.X[permutation]
        self.y = self.y[permutation]

    def add_noise(self, fraction):
        assert fraction >= 0.0 and fraction <= 1.0, "Invalid fraction specified"
        self.permute_data()
        split_index = self.size * fraction
        self.y[:split_index] = self.y[:split_index] ^ 1
        self.permute_data()

    def train_val_test_split(self, train_fraction=None, val_fraction=None, train_size=None, val_size=None):
        assert (train_fraction is not None and val_fraction is not None) or (train_size is not None and val_size is not None), "Split improperly specified"
        if train_fraction is None:
            assert train_size >= 0 and val_size >= 0 and train_size + val_size <= self.size, "Invalid size"
            train_split_index = train_size
            val_split_index = train_size + val_size
        else:
            assert train_fraction >= 0.0 and val_fraction >= 0.0 and train_fraction + val_fraction <= 1.0, "Invalid fractions"
            train_split_index = self.size * train_fraction
            val_split_index = self.size * (train_fraction + val_fraction)
        permutation = np.random.permutation(np.arange(self.size))
        self.train_indices = permutation[:train_split_index]
        self.val_indices = permutation[train_split_index:val_split_index]
        self.test_indices = permutation[val_split_index:]
        self.train_X = self.X[self.train_indices]
        self.train_y = self.y[self.train_indices]
        self.val_X = self.X[self.val_indices]
        self.val_y = self.y[self.val_indices]
        self.test_X = self.X[self.test_indices]
        self.test_y = self.y[self.test_indices]
