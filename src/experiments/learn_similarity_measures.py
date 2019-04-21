import kernels
import numpy as np
import utils

from ..datasets import random_dataset

def train(data_loader, kernel, reg_param, width):
    gram_matrix = utils.generate_gram_matrix(data_loader.train_X, kernel, width)
    weights = utils.solve(gram_matrix, data_loader.train_y, reg_param)
    def h(edge):
        sum = 0.0
        for i in range(len(weights)):
            sum += weights[i] * kernel(data_loader.train_X[i], edge)
        return sum
    return h

def load_data(size, num_nodes, dims, similarity_vector, p=0.5, dist_func=bernoulli, noise=0.1, train_frac=None, val_frac=None, train_size=500, val_size=500):
    data_loader = random_dataset.RandomDataset(size, num_nodes, dims, similarity_vector, p, dist_func)
    data_loader.add_noise(noise)
    data_loader.train_val_test_split(train_frac, val_frac, train_size, val_size)
    return data_loader

def run(args):
    data_loader = load_data(args.size, args.num_nodes, args.dims, args.sim_vec, args.p, args.dist_func, args.noise, args.train_frac, args.val_frac, args.train_size, args.val_size)
    kernel = utils.map_kernel(args.kernel)
    h = train(data_loader, kernel, args.reg_param, args.width)
    sigma = utils.map_sigma(args.sigma)
    predictor = utils.get_predictor(h, sigma, args.b)
