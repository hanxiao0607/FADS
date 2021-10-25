from argparse import ArgumentParser

from IDS.utils import utils
from IDS.model import rad

import warnings
warnings.filterwarnings('ignore')


def arg_parser():
    """
    Add parser parameters
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument('--random_seed', help='random seed', default=9)
    parser.add_argument('--dataset_dir', help='please choose dataset directory', default='./Datasets/IDS2018_small.csv')
    parser.add_argument('--out_dim', help='output dimensions', default=128)
    parser.add_argument('--lr', help='learning rate', default=0.001)
    parser.add_argument('--device', help='device cpu or cuda', default='cuda:0')
    parser.add_argument('--dataset', help='name of dataset', default='IDS2018_small')
    parser.add_argument('--r_alpha', help='hyper-parameter for the reward of anomaly detection', default=0.1)
    parser.add_argument('--r_ad_alpha', help='hyper-parameter for the reward of anomaly detection', default=0)
    parser.add_argument('--r_cl_alpha', help='hyper-parameter for the reward of anomaly classification', default=1)

    # set unseen validation set parameters
    parser.add_argument('--validation_size', help='set validation size for each class', default=10)

    # set classifier network parameters
    parser.add_argument('--input_dim', help='input dimensions', default=77)
    parser.add_argument('--hid_dim0', default=256)
    parser.add_argument('--hid_dim1', default=64)
    parser.add_argument('--n_ways', help='n ways', default=4)
    parser.add_argument('--n_support', help='n support', default=3)
    parser.add_argument('--n_query', help='n query', default=3)
    parser.add_argument('--max_epoch', help='max epoch for prototypical networks', default=10)
    parser.add_argument('--epoch_size', help='epoch size for each epoch of protonet', default=100)

    # set RAD parameters
    parser.add_argument('--max_episode', help='max episode for each iterators', default=100)
    parser.add_argument('--max_iterators', help='max iterators for training RAD model', default=10)
    parser.add_argument('--lambda', help='hyper-parameter to balance two rewards', default=1)
    parser.add_argument('--num_samples', help='number of samples generated each episode', default=5)
    parser.add_argument('--n_min_size', help='minimum number of final training size', default=10)
    parser.add_argument('--ratio_ab', help='ratio of abnormal', default=0.1)

    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()
    options = vars(args)

    utils.set_seed(options['random_seed'])

    # splitting dataset
    seen_x, seen_y, sup_x, sup_y, unseen_x, unseen_y, test_x, test_y = utils.preprocessing_IDS(options)

    df_seen = utils.data2df(seen_x, seen_y)
    df_sup = utils.data2df(sup_x, sup_y)
    df_unseen = utils.data2df(unseen_x, unseen_y)

    model_rad = rad.RAD(options)
    model_rad.train_rad(df_seen, df_unseen, df_sup, test_x, test_y)

    print('done')


if __name__ == '__main__':
    main()
