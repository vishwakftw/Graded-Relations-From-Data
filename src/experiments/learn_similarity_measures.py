import kernels
import math
import numpy as np
import utils

from ..datasets import random_dataset
from ..distributions import bernoulli
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import PredefinedSplit


def load_data(args):
    dist_func = utils.map_dist_func(args.dist_func)
    data_loader = random_dataset.RandomDataset(args.size, args.num_nodes, args.dims, args.similarity_vector, args.p, dist_func)
    data_loader.add_noise(args.noise)
    data_loader.train_val_test_split(args.train_frac, args.val_frac, args.train_size, args.val_size)
    return data_loader


class LearnSimilarityMeasures(BaseEstimator, RegressorMixin):

    def __init__(self, kernel_name='cartesian', sigma_name='sigmoid', b=1.0, reg_param=1.0, width=1.0):
        self.kernel_name = kernel_name
        self.sigma_name = sigma_name
        self.b = b
        self.reg_param = reg_param
        self.width = width

    def fit(self, X, y):
        kernel = utils.map_kernel(self.kernel_name)
        sigma = utils.map_sigma(self.sigma_name)
        gram_matrix = utils.generate_gram_matrix(X, kernel, self.width)
        weights = utils.solver(gram_matrix, y, self.reg_param)
        def h(edge):
            sum = 0.0
            for i in range(len(weights)):
                sum += weights[i] * kernel(X[i], edge)
            return sum
        self.predictor_ = utils.get_predictor(h, sigma, self.b)
        return self

    def predict(self, X):
        predictions = np.array(list(map(self.predictor, X)))
        return predictions


def run(args):
    param_grid = {'kernel_name': ['cartesian'], 'sigma_name': ['sigmoid'], 'b': [1.0, 2.0], 'reg_param': [math.pow(x-20) for x in range(22)], 'width': [math.pow(x-20) for x in range(22)]}
    data_loader = load_data(args)
    test_fold = np.full(data_loader.size, -1)
    test_fold[data_loader.test_indices] = 0
    ps = PredefinedSplit(test_fold)
    model = LearnSimilarityMeasures(args.kernel, args.sigma)
    cv = GridSearchCV(model, param_grid=param_grid, scoring=mean_squared_error, cv=ps)
    cv.fit(data_loader.X)
    predictions = cv.predict(data_loader.test_X)
    error = mean_squared_error(predictions, data_loader.test_y)
    print("Error: ",error)
    
