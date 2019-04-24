import numpy as np
import grd.utils as utils


class SpeciesCompetitionDataset(object):
    def __init__(self, train_size, val_size, test_size, num_train_species,
                 num_val_species, num_test_species, k):
        assert train_size > 0 and val_size > 0 and test_size > 0 \
            and num_train_species > 0 and num_val_species > 0 and num_test_species > 0 \
            and k > 0, "Invalid size, n, or k specified"
        self.train_size = num_train_species
        self.val_size = num_val_species
        self.test_size = num_test_species
        self.k = k
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.train_nodes = np.random.uniform(size=(num_train_species, k))
        self.val_nodes = np.random.uniform(size=(num_val_species, k))
        self.test_nodes = np.random.uniform(size=(num_test_species, k))
        self.train_X, self.train_y = self.generate_data(self.train_nodes, self.train_size)
        self.val_X, self.val_y = self.generate_data(self.val_nodes, self.val_size)
        self.test_X, self.test_y = self.generate_data(self.test_nodes, self.test_size)

    def generate_data(self, nodes, size):
        pairs = np.array([(x, y) for x in nodes for y in nodes if np.any(x != y)])
        edges = np.random.permutation(pairs)[:size]
        labels = np.array([utils.heaviside_similarity(e) for e in edges])
        return edges, labels

    def cv_x(self):
        return np.vstack((self.train_X, self.val_X))

    def cv_y(self):
        return np.hstack((self.train_y, self.val_y))

    def cv_size(self):
        return self.train_X.shape[0] + self.val_X.shape[0]

    def cv_indices(self):
        return np.arange(self.train_X.shape[0], self.train_X.shape[0] + self.val_X.shape[0])
