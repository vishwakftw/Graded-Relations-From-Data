import numpy as np
import utils


class SpeciesCompetitionDataset(object):

    def __init__(self, size, num_species, k):
        assert size > 0 and num_species > 0 and k > 0, "Invalid size, n, or k specified"
        self.n = num_species
        self.k = k
        self.size = size
        self.nodes = np.random.uniform(size=(num_species, k))
        self.X, self.y = self.generate_data(self.nodes, size)

    def generate_data(self, nodes, size):
        pairs = np.array([(x, y) for x in nodes for y in nodes if x != y])
        edges = np.random.permutation(pairs)[:size]
        labels = np.array([utils.heaviside_similarity(e) for e in edges])
        return edges, labels