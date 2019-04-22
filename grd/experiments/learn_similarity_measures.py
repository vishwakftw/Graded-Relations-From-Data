import grd.kernels as kernels
import math
import numpy as np
import grd.utils as utils

from ..datasets import random_dataset
from ..distributions import bernoulli
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import PredefinedSplit


def load_data(args):
    dist_func = utils.map_dist_func(args.dist_func)
    similarity_vector = utils.map_similarity_vector(args.similarity_vector)
    data_loader = random_dataset.RandomDataset(args.size, args.num_nodes, args.dims, similarity_vector, args.p, dist_func)
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
        gram_matrix = utils.generate_gram_matrix(X, kernel, width=self.width)
        weights = utils.solver(gram_matrix, y, self.reg_param)
        def h(edge):
            sum = 0.0
            for i in range(len(weights)):
                sum += weights[i] * kernel(X[i], edge)
            return sum
        self.predictor_ = utils.get_predictor(h, sigma, self.b)
        return self

    def predict(self, X):
        predictions = np.array(list(map(self.predictor_, X)))
        return predictions


def run(args):
    param_grid = {'kernel_name': ['cartesian'], 'sigma_name': ['sigmoid'], 'b': [1.0], 'reg_param': [math.pow(2,x-2) for x in range(2)], 'width': [math.pow(2,x-2) for x in range(2)]}
    data_loader = load_data(args)
    print("Generated data")
    test_fold = np.full(data_loader.cv_size(), -1)
    test_fold[data_loader.cv_indices()] = 0
    ps = PredefinedSplit(test_fold)
    model = LearnSimilarityMeasures(args.kernel, args.sigma)
    print("Created model")
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    cv = GridSearchCV(model, param_grid=param_grid, scoring=scorer, cv=ps)
    print("About to fit")
    cv.fit(data_loader.cv_x(), data_loader.cv_y())
    # model.fit(data_loader.train_X, data_loader.train_y)
    print("Done fitting")
    print(data_loader.test_X)
    predictions = cv.predict(data_loader.test_X)
    error = mean_squared_error(predictions, data_loader.test_y)
    print("Error: ",error)
    
