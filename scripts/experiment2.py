from grd.experiments.learn_similarity_measures import run

from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="Run experiments in Section C of the paper")
    parser.add_argument('--num_train_species', type=int, required=True,
                        help='Number of species in the training graph')
    parser.add_argument('--num_val_species', type=int, required=True,
                        help='Number of species in the validation graph')
    parser.add_argument('--num_test_species', type=int, required=True,
                        help='Number of species in the testing graph')
    parser.add_argument('--k', type=int, required=True,
                        help='Number of limiting factors')
    parser.add_argument('--kernel', type=str, required=True,
                        choices=['cartesian', 'kronecker', 'reciprocal_kronecker',
                                 'symmetric_kronecker', 'mlpk'], help='Kernel to use')
    parser.add_argument('--train_size', type=int, required=True,
                        help='Number of data points to be used for training')
    parser.add_argument('--val_size', type=int, required=True,
                        help='Number of data points to be used for validation')
    parser.add_argument('--test_size', type=int, required=True,
                        help='Number of data points to be used for testing')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--jobs', default=-1, type=int, help='Number of jobs for grid search')

    args = parser.parse_args()
    args.exp = 2
    run(args)


if __name__ == "__main__":
    main()
