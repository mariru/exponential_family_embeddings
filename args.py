import argparse
import os
import time
def parse_args():
        parser = argparse.ArgumentParser(description="run exponential family embeddings on text")

        parser.add_argument('--K', type=int, default=100,
                            help='Number of dimensions. Default is 100.')

        parser.add_argument('--L', type=int, default=50000,
                            help='Vocabulary size. Default is 50000.')

        parser.add_argument('--n_iter', type=int, default = 100,
                            help='Number iterations. Default is 500.')

        parser.add_argument('--n_epochs', type=int, default=10,
                            help='Number of epochs. Default is 10.')

        parser.add_argument('--cs', type=int, default=4,
                            help='Context size. Default is 4.')

        parser.add_argument('--ns', type=int, default=100,
                            help='Number of negative samples. Default is 100.')

        parser.add_argument('--mb', type=int, default=5000,
                            help='Minibatch size. Default is 5000.')

        args =  parser.parse_args()
        dir_name = 'fits/fit' + time.strftime("%y_%m_%d_%H_%M_%S")

        return args, dir_name
