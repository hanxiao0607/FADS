from models import utils, OCSVM


def main():
    dir_UNSW = '../UNSW/Datasets/NUSW_small.csv'
    lst = list(range(10))
    print('UNSW')
    for i in lst:
        seen_x, seen_y, test_x, test_y = utils.preprocessing_UNSW(dir_UNSW, n_sup=10, seed=i, adc=0, unsupervised=1)
        OCSVM.SVM(seen_x, seen_y, test_x, test_y)
    dir_IDS = '../IDS/Datasets/IDS2018_small.csv'
    print('-'*20)
    print('IDS')
    for i in lst:
        seen_x, seen_y, test_x, test_y = utils.preprocessing_IDS(dir_IDS, n_sup=10, seed=i, adc=0, unsupervised=1)
        OCSVM.SVM(seen_x, seen_y, test_x, test_y)
    dir_CERT = '../CERT/Datasets/CERT52_small.csv'
    print('-'*20)
    print('CERT')
    for i in lst:
        seen_x, seen_y, test_x, test_y = utils.preprocessing_CERT_EMB(dir_CERT, n_sup=10, seed=i, adc=0, unsupervised=1)
        OCSVM.SVM(seen_x, seen_y, test_x, test_y)
    print('done')


if __name__ == '__main__':
    main()
