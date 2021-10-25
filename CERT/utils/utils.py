import random
import numpy as np
import torch
import pandas as pd
import warnings
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm


warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_dataset(dir):
    return pd.read_csv(dir, index_col=0)


def get_cat_list(df):
    lst_cat = []
    lst_num = []
    for ind, t in enumerate(df.dtypes):
        if t == 'object':
            lst_cat.append(df.columns[ind])
        else:
            lst_num.append(df.columns[ind])
    return lst_cat, lst_num


def sess_window(df):
    return df.groupby(['sess_id']).agg({'feature': list, 'label': max})


def get_training_dictionary(df):
    dic = {'pad': 0}
    count = 1
    for i in range(len(df)):
        lst = list(df['feature'].iloc[i])
        for j in lst:
            if j in dic:
                pass
            else:
                dic[j] = count
                count += 1
    return dic


def key2num(ary, dic):
    new_ary = []
    for ind, val in enumerate(ary):
        temp = []
        for j in val[0]:
            if j in dic:
                temp.append(dic[j])
            else:
                temp.append(len(dic))
        new_ary.append(temp)
    return new_ary

def preprocessing_CERT_EMB(options, n_class=4):
    dir = options['dataset_dir']
    n_sup = options['validation_size']
    seed = options['random_seed']
    df = load_dataset(dir)
    df_sess = sess_window(df)
    df0 = df_sess.loc[df_sess['label'] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
    df0['length'] = df0.feature.map(len)
    df0 = df0.loc[df0['length'] <= 134].reset_index(drop=True)
    df0.drop(columns='length', inplace=True)
    df1 = df_sess.loc[df_sess['label'] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    df1['length'] = df1.feature.map(len)
    df1 = df1.loc[df1['length'] <= 134].reset_index(drop=True)
    df1.drop(columns='length', inplace=True)
    df2 = df_sess.loc[df_sess['label'] == 2].sample(frac=1, random_state=seed).reset_index(drop=True)
    df2['length'] = df2.feature.map(len)
    df2 = df2.loc[df2['length'] <= 134].reset_index(drop=True)
    df2.drop(columns='length', inplace=True)
    df3 = df_sess.loc[df_sess['label'] == 3].sample(frac=1, random_state=seed).reset_index(drop=True)
    df3['length'] = df3.feature.map(len)
    df3 = df3.loc[df3['length'] <= 134].reset_index(drop=True)
    df3.drop(columns='length', inplace=True)
    df4 = df_sess.loc[df_sess['label'] == 4].sample(frac=1, random_state=seed).reset_index(drop=True)
    df4['length'] = df4.feature.map(len)
    df4 = df4.loc[df4['length'] <= 134].reset_index(drop=True)
    df4.drop(columns='length', inplace=True)

    lst = [df0, df1, df2, df3, df4]
    print(f'samples for each classes: {[len(i) for i in lst]}')
    dic = get_training_dictionary(df0[:100])
    print(f'Training keys: {len(dic)}')
    print(f'Total keys: {len(set(df["feature"]))}')

    seen_x = key2num(df0.iloc[:100, :-1].values, dic)
    seen_y = df0.iloc[:100, -1].values
    if n_class == 4:
        sup_x = key2num(pd.concat([df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
        sup_y = pd.concat([df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, -1].values
        unseen_x = key2num(pd.concat([df0.iloc[20000:40000], df1.iloc[n_sup:28+n_sup], df2.iloc[n_sup:201+n_sup], df3.iloc[n_sup:10+n_sup], df4.iloc[n_sup:21+n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
        unseen_y = pd.concat([df0.iloc[20000:40000], df1.iloc[n_sup:28+n_sup], df2.iloc[n_sup:201+n_sup], df3.iloc[n_sup:10+n_sup], df4.iloc[n_sup:21+n_sup]], axis=0, ignore_index=True).iloc[:, -1].values
        test_x = key2num(pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
        test_y = pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0, ignore_index=True).iloc[:, -1].values
    elif n_class == 3:
        sup_x = key2num(pd.concat([df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup]], axis=0,
                                  ignore_index=True).iloc[:, :-1].values, dic)
        sup_y = pd.concat([df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup]], axis=0,
                          ignore_index=True).iloc[:, -1].values
        unseen_x = key2num(pd.concat(
            [df0.iloc[20000:40000], df1.iloc[n_sup:28 + n_sup], df2.iloc[n_sup:201 + n_sup], df3.iloc[n_sup:10 + n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
        unseen_y = pd.concat(
            [df0.iloc[20000:40000], df1.iloc[n_sup:28 + n_sup], df2.iloc[n_sup:201 + n_sup], df3.iloc[n_sup:10 + n_sup]], axis=0, ignore_index=True).iloc[:, -1].values
        test_x = key2num(
            pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:]], axis=0,
                      ignore_index=True).iloc[:, :-1].values, dic)
        test_y = pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:]], axis=0,
                           ignore_index=True).iloc[:, -1].values
    elif n_class == 2:
        sup_x = key2num(pd.concat([df1.iloc[:n_sup], df2.iloc[:n_sup]], axis=0,
                                  ignore_index=True).iloc[:, :-1].values, dic)
        sup_y = pd.concat([df1.iloc[:n_sup], df2.iloc[:n_sup]], axis=0,
                          ignore_index=True).iloc[:, -1].values
        unseen_x = key2num(pd.concat(
            [df0.iloc[20000:40000], df1.iloc[n_sup:28 + n_sup], df2.iloc[n_sup:201 + n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
        unseen_y = pd.concat(
            [df0.iloc[20000:40000], df1.iloc[n_sup:28 + n_sup], df2.iloc[n_sup:201 + n_sup]], axis=0, ignore_index=True).iloc[:, -1].values
        test_x = key2num(
            pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:]], axis=0,
                      ignore_index=True).iloc[:, :-1].values, dic)
        test_y = pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:]], axis=0,
                           ignore_index=True).iloc[:, -1].values
    elif n_class == 1:
        sup_x = key2num(df1.iloc[:n_sup].iloc[:, :-1].values, dic)
        sup_y = df1.iloc[:n_sup].iloc[:, -1].values
        unseen_x = key2num(pd.concat(
            [df0.iloc[20000:4000], df1.iloc[n_sup:28 + n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
        unseen_y = pd.concat(
            [df0.iloc[20000:40000], df1.iloc[n_sup:28 + n_sup]], axis=0, ignore_index=True).iloc[:, -1].values
        test_x = key2num(
            pd.concat([df0.iloc[-2000:], df1.iloc[-27:]], axis=0,
                      ignore_index=True).iloc[:, :-1].values, dic)
        test_y = pd.concat([df0.iloc[-2000:], df1.iloc[-27:]], axis=0,
                           ignore_index=True).iloc[:, -1].values

    return seen_x, seen_y, sup_x, sup_y, unseen_x, unseen_y, test_x, test_y, len(dic)+1


def data2df(x, y):
    df = pd.DataFrame()
    df['sdata'] = x
    df['y_true'] = y
    df['y_pred'] = y
    df['dist'] = 0
    return df
