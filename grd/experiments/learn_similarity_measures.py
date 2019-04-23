import numpy as np
import grd.utils as utils

from ..datasets import random_dataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import PredefinedSplit


def load_data(args):
    dist_func = utils.map_dist_func(args.dist_func)
    similarity_vector = utils.map_similarity_vector(args.similarity_vector)
    data_loader = random_dataset.RandomDataset(args.size, args.num_nodes,
                                               args.dims, similarity_vector,
                                               args.p, dist_func, not args.noshare_nodes)
    data_loader.add_noise(args.noise)
    if not args.noshare_nodes:
        data_loader.train_val_test_split(args.train_frac, args.val_frac,
                                         args.train_size, args.val_size)
    return data_loader


class LearnSimilarityMeasures(BaseEstimator, RegressorMixin):

    def __init__(self, kernel_name='cartesian', reg_param=1.0, width=1.0):
        self.kernel_name = kernel_name
        self.reg_param = reg_param
        self.width = width

    def fit(self, X, y):
        kernel = utils.map_kernel(self.kernel_name)
        gram_matrix = utils.generate_gram_matrix(X, kernel, width=self.width)
        weights = utils.solver(gram_matrix, y, self.reg_param)

        def h(edge):
            sum = 0.0
            for i in range(len(weights)):
                sum += weights[i] * kernel(X[i], edge, width=self.width)
            return sum

        # we clamp so that the values are in [0, 1]
        self.predictor_ = lambda x: max(0., min(1., h(x)))
        return self

    def predict(self, X):
        predictions = np.array(list(map(self.predictor_, X)))
        return predictions


def run(args):
    np.random.seed(args.seed)
    param_grid = {'kernel_name': [args.kernel], 'reg_param': 2. ** np.arange(-20, 2, 1),
                  'width': 2. ** np.arange(-20, 2, 1)}
    print("Generating data....")
    data_loader = load_data(args)
    print("Done.")

    print("Creating model....")
    test_fold = np.full(data_loader.cv_size(), -1)
    test_fold[data_loader.cv_indices()] = 0
    ps = PredefinedSplit(test_fold)
    model = LearnSimilarityMeasures(args.kernel)
    print("Done.")

    print("Initializing grid for grid search....")
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    cv = GridSearchCV(model, param_grid=param_grid, scoring=scorer, cv=ps, n_jobs=args.jobs)
    print("Done.")

    print("Fitting....")
    cv.fit(data_loader.cv_x(), data_loader.cv_y())
    print("Done.")

    print("Generating predictions....")
    predictions = cv.predict(data_loader.test_X)
    error = mean_squared_error(predictions, data_loader.test_y)
    print("Error: ", error)
    print("Baseline Error (mean prediction): ",
          mean_squared_error(np.full(predictions.shape[0], np.mean(data_loader.train_y)),
                             data_loader.test_y))
    print("Done.")
