from grd.experiments.learn_similarity_measures import run

from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="Run experiments in Section A of the paper")
    parser.add_argument('--size', type=int, required=True,
                        help='Size of the dataset, i.e., \
                              number of edges (train + validation + test)')
    parser.add_argument('--num_nodes', type=int, required=True,
                        help='Number of nodes in the graph')
    parser.add_argument('--dims', type=int, required=True,
                        help='Number of dimensions in the feature vector')
    parser.add_argument('--similarity_vector', type=str, choices=['jaccard', 'sokal', 'tp'],
                        required=True, help='Similarity metric to be used for \
                                             constructing the dataset')
    parser.add_argument('--dist_func', default='bernoulli', type=str, choices=['bernoulli'],
                        required=False, help='Distribution to sample for generating the dataset')
    parser.add_argument('--p', default=0.5, type=float, required=False,
                        help='Parameter of Bernoulli distribution for \
                              generating the dataset')
    parser.add_argument('--noise', default=0.1, type=float, required=False,
                        help='Fraction of noise to add to the generated dataset')
    parser.add_argument('--kernel', type=str, required=True,
                        choices=['cartesian', 'kronecker', 'reciprocal_kronecker',
                                 'symmetric_kronecker', 'mlpk'], help='Kernel to use')
    parser.add_argument('--train_frac', default=None, type=float, required=False,
                        help='Fraction of dataset to be used for training')
    parser.add_argument('--val_frac', default=None, type=float, required=False,
                        help='Fraction of dataset to be used for validation')
    parser.add_argument('--train_size', default=None, type=int, required=False,
                        help='Number of data points to be used for training')
    parser.add_argument('--val_size', default=None, type=int, required=False,
                        help='Number of data points to be used for validation')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--jobs', default=-1, type=int, help='Number of jobs for grid search')
    parser.add_argument('--share_nodes', action='store_true',
                        help='Whether train, validation, and test sets use the same graph nodes')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
