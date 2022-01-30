import random
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE


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

def preprocessing_UNSW(dir, n_sup=10, seed=42, adc=0, unsupervised=1, oversampling=0):
    df = load_dataset(dir)
    scaler = MinMaxScaler()
    cat_col, num_col = get_cat_list(df)
    df[num_col[:-2]] = scaler.fit_transform(df[num_col[:-2]])
    df_pre = pd.concat([pd.get_dummies(df[cat_col]), df[num_col[:-2]], df[num_col[-2:]]], axis=1)
    df_ohe = df_pre.apply(pd.to_numeric)
    df_ohe = df_ohe.drop(columns=['Label'])
    df0 = df_ohe.loc[df_ohe['attack_cat'] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
    df1 = df_ohe.loc[df_ohe['attack_cat'] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    df2 = df_ohe.loc[df_ohe['attack_cat'] == 2].sample(frac=1, random_state=seed).reset_index(drop=True)
    df3 = df_ohe.loc[df_ohe['attack_cat'] == 3].sample(frac=1, random_state=seed).reset_index(drop=True)
    df4 = df_ohe.loc[df_ohe['attack_cat'] == 4].sample(frac=1, random_state=seed).reset_index(drop=True)
    df5 = df_ohe.loc[df_ohe['attack_cat'] == 5].sample(frac=1, random_state=seed).reset_index(drop=True)
    df6 = df_ohe.loc[df_ohe['attack_cat'] == 6].sample(frac=1, random_state=seed).reset_index(drop=True)
    df7 = df_ohe.loc[df_ohe['attack_cat'] == 7].sample(frac=1, random_state=seed).reset_index(drop=True)
    df4['attack_cat'] = 2
    df5['attack_cat'] = 3
    df6['attack_cat'] = 4
    df7['attack_cat'] = 5
    if unsupervised == 1:
        seen_x = df0.iloc[:20000].iloc[:, :-1].values
        seen_y = df0.iloc[:20000].iloc[:, -1].values
        test_x = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:], df6.iloc[-2000:],
                            df7.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, :-1].values
        test_y = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:], df6.iloc[-2000:],
                            df7.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, -1].values
        if adc ==0:
            test_y[20000:] = 1
    else:
        seen_x = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df4.iloc[:n_sup], df5.iloc[:n_sup], df6.iloc[:n_sup], df7.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values
        seen_y = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df4.iloc[:n_sup], df5.iloc[:n_sup], df6.iloc[:n_sup], df7.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, -1].values
        test_x = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:], df6.iloc[-2000:], df7.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, :-1].values
        test_y = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:], df6.iloc[-2000:], df7.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, -1].values
        if adc == 0:
            seen_y[20000:] = 1
            test_y[20000:] = 1
        if oversampling == 1:
            sm = SMOTE(random_state=seed)
            df_temp = df0.iloc[:20000]
            df_temp = pd.concat([df_temp, df1.iloc[:n_sup], df4.iloc[:n_sup], df5.iloc[:n_sup], df6.iloc[:n_sup],
                 df7.iloc[:n_sup]], axis=0, ignore_index=True)
            seen_x = df_temp.iloc[:, :-1].values
            seen_y = df_temp.iloc[:, -1].values
            seen_x, seen_y = sm.fit_resample(seen_x, seen_y)
    return seen_x, seen_y, test_x, test_y

def preprocessing_UNSW_DeepSAD(dir, n_sup=10, seed=42, adc=0, pul=0):
    df = load_dataset(dir)
    scaler = MinMaxScaler()
    cat_col, num_col = get_cat_list(df)
    df[num_col[:-2]] = scaler.fit_transform(df[num_col[:-2]])
    df_pre = pd.concat([pd.get_dummies(df[cat_col]), df[num_col[:-2]], df[num_col[-2:]]], axis=1)
    df_ohe = df_pre.apply(pd.to_numeric)
    df_ohe = df_ohe.drop(columns=['Label'])
    df0 = df_ohe.loc[df_ohe['attack_cat'] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
    df1 = df_ohe.loc[df_ohe['attack_cat'] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    df2 = df_ohe.loc[df_ohe['attack_cat'] == 2].sample(frac=1, random_state=seed).reset_index(drop=True)
    df3 = df_ohe.loc[df_ohe['attack_cat'] == 3].sample(frac=1, random_state=seed).reset_index(drop=True)
    df4 = df_ohe.loc[df_ohe['attack_cat'] == 4].sample(frac=1, random_state=seed).reset_index(drop=True)
    df5 = df_ohe.loc[df_ohe['attack_cat'] == 5].sample(frac=1, random_state=seed).reset_index(drop=True)
    df6 = df_ohe.loc[df_ohe['attack_cat'] == 6].sample(frac=1, random_state=seed).reset_index(drop=True)
    df7 = df_ohe.loc[df_ohe['attack_cat'] == 7].sample(frac=1, random_state=seed).reset_index(drop=True)
    df4['attack_cat'] = 2
    df5['attack_cat'] = 3
    df6['attack_cat'] = 4
    df7['attack_cat'] = 5
    seen_x = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df4.iloc[:n_sup], df5.iloc[:n_sup], df6.iloc[:n_sup], df7.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values
    seen_y = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df4.iloc[:n_sup], df5.iloc[:n_sup], df6.iloc[:n_sup], df7.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, -1].values
    unseen_x = df0.iloc[20000:50000].iloc[:, :-1].values
    test_x = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:], df6.iloc[-2000:], df7.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, :-1].values
    test_y = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:], df6.iloc[-2000:], df7.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, -1].values
    if adc == 0:
        seen_y[20000:] = 1
        test_y[20000:] = 1
    if pul == 0:
        return seen_x, seen_y, unseen_x, test_x, test_y
    else:
        X = np.concatenate([seen_x, unseen_x], axis=0)
        seen_y = np.where(seen_y > 0, 1, seen_y)
        y = np.concatenate([seen_y, [0 for _ in range(len(unseen_x))]])
        return X, y, test_x, test_y

def preprocessing_IDS(dir, n_sup=10, seed=42, adc=0, unsupervised=1, oversampling=0):
    df = load_dataset(dir)
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    df0 = df.loc[df['Label'] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
    df1 = df.loc[df['Label'] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    df['Label'].loc[df['Label'] == 3] = 2
    df2 = df.loc[df['Label'] == 2].sample(frac=1, random_state=seed).reset_index(drop=True)
    df['Label'].loc[df['Label'] == 5] = 4
    df['Label'].loc[df['Label'] == 4] = 3
    df3 = df.loc[df['Label'] == 3].sample(frac=1, random_state=seed).reset_index(drop=True)
    if unsupervised == 1:
        seen_x = df0.iloc[:20000].iloc[:, :-1].values
        seen_y = df0.iloc[:20000].iloc[:, -1].values
        test_x = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df2.iloc[-2000:], df3.iloc[-2000:]], axis=0,
                           ignore_index=True).iloc[:, :-1].values
        test_y = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df2.iloc[-2000:], df3.iloc[-2000:]], axis=0,
                           ignore_index=True).iloc[:, -1].values
        if adc == 0:
            test_y[20000:] = 1
    else:
        seen_x = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values
        seen_y = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, -1].values
        test_x = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df2.iloc[-2000:], df3.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, :-1].values
        test_y = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df2.iloc[-2000:], df3.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, -1].values
        if adc == 0:
            seen_y[20000:] = 1
            test_y[20000:] = 1
        if oversampling == 1:
            seen_x = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup]], axis=0,
                               ignore_index=True).iloc[:, :-1].values
            seen_y = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup]], axis=0,
                               ignore_index=True).iloc[:, -1].values

            sm = SMOTE(random_state=seed)
            seen_x, seen_y = sm.fit_resample(seen_x, seen_y)
    return seen_x, seen_y, test_x, test_y


def preprocessing_IDS_DeepSAD(dir, n_sup=10, seed=42, adc=0, pul=0):
    df = load_dataset(dir)
    scaler = MinMaxScaler()
    df.iloc[:, :-1] = scaler.fit_transform(df.iloc[:, :-1])
    df0 = df.loc[df['Label'] == 0].sample(frac=1, random_state=seed).reset_index(drop=True)
    df1 = df.loc[df['Label'] == 1].sample(frac=1, random_state=seed).reset_index(drop=True)
    df['Label'].loc[df['Label'] == 3] = 2
    df2 = df.loc[df['Label'] == 2].sample(frac=1, random_state=seed).reset_index(drop=True)
    df['Label'].loc[df['Label'] == 5] = 4
    df['Label'].loc[df['Label'] == 4] = 3
    df3 = df.loc[df['Label'] == 3].sample(frac=1, random_state=seed).reset_index(drop=True)
    seen_x = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values
    seen_y = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, -1].values
    unseen_x = df0.iloc[20000:50000].iloc[:, :-1].values
    test_x = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df2.iloc[-2000:], df3.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, :-1].values
    test_y = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df2.iloc[-2000:], df3.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, -1].values
    if adc == 0:
        seen_y[20000:] = 1
        test_y[20000:] = 1
    if pul == 0:
        return seen_x, seen_y, unseen_x, test_x, test_y
    else:
        X = np.concatenate([seen_x, unseen_x], axis=0)
        seen_y = np.where(seen_y > 0, 1, seen_y)
        y = np.concatenate([seen_y, [0 for _ in range(len(unseen_x))]])
        return X, y, test_x, test_y


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
        new_ary.append(str(temp))


    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(new_ary)
    return X.toarray()


def preprocessing_CERT_EMB(dir, n_sup=10, seed=42, adc=0, unsupervised=1, oversampling=0):
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

    dic = get_training_dictionary(pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True))

    if unsupervised == 1:
        seen_x = key2num(df0.iloc[:20000].iloc[:, :-1].values, dic)
        seen_y = df0.iloc[:20000].iloc[:, -1].values

        test_x = key2num(
            pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0,
                      ignore_index=True).iloc[:, :-1].values, dic)
        test_y = pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0,
                           ignore_index=True).iloc[:, -1].values
        if adc == 0:
            test_y[2000:] = 1

    else:
        seen_x = key2num(pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
        seen_y = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, -1].values

        test_x = key2num(pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
        test_y = pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0, ignore_index=True).iloc[:, -1].values

        if adc == 0:
            seen_y[20000:] = 1
            test_y[2000:] = 1
        if oversampling == 1:
            sm = SMOTE(random_state=seed)
            df_temp = df0.iloc[:20000]
            df_temp = pd.concat([df_temp, df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True)
            seen_x = key2num(df_temp.iloc[:, :-1].values, dic)
            seen_y = df_temp.iloc[:, -1].values
            seen_x, seen_y = sm.fit_resample(seen_x, seen_y)
    return seen_x, seen_y, test_x, test_y


def key2num_DeepSVDD(ary, dic):
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


def preprocessing_CERT_EMB_DeepSVDD(dir, n_sup=10, seed=42, adc=0, unsupervised=1):
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

    dic = get_training_dictionary(pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True))

    if unsupervised == 1:
        seen_x = key2num_DeepSVDD(df0.iloc[:20000].iloc[:, :-1].values, dic)
        seen_y = df0.iloc[:20000].iloc[:, -1].values

        test_x = key2num_DeepSVDD(
            pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0,
                      ignore_index=True).iloc[:, :-1].values, dic)
        test_y = pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0,
                           ignore_index=True).iloc[:, -1].values
        if adc == 0:
            test_y[2000:] = 1

    else:
        seen_x = key2num_DeepSVDD(pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
        seen_y = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, -1].values

        test_x = key2num_DeepSVDD(pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
        test_y = pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0, ignore_index=True).iloc[:, -1].values

        if adc == 0:
            seen_y[20000:] = 1
            test_y[2000:] = 1

    return seen_x, seen_y, test_x, test_y


def preprocessing_CERT_EMB_DeepSAD(dir, n_sup=10, seed=42, adc=0, unsupervised=1):
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

    dic = get_training_dictionary(pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True))

    seen_x = key2num_DeepSVDD(pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
    seen_y = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, -1].values

    unseen_x = key2num_DeepSVDD(df0.iloc[20000:40000].iloc[:, :-1].values, dic)

    test_x = key2num_DeepSVDD(pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0, ignore_index=True).iloc[:, :-1].values, dic)
    test_y = pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0, ignore_index=True).iloc[:, -1].values

    if adc == 0:
        seen_y[20000:] = 1
        test_y[2000:] = 1

    if unsupervised == 0:
        df_temp = df0.iloc[:20000]
        for i in range(20000//n_sup):
            df_temp = pd.concat([df_temp, df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0, ignore_index=True)
        seen_x = key2num_DeepSVDD(df_temp.iloc[:, :-1].values, dic)
        seen_y = df_temp.iloc[:, -1].values

    return seen_x, seen_y, unseen_x, test_x, test_y


def preprocessing_CERT_EMB_PUL(dir, n_sup=10, seed=42):
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

    dic = get_training_dictionary(
        pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0,
                  ignore_index=True))

    seen_x = key2num_DeepSVDD(
        pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]], axis=0,
                  ignore_index=True).iloc[:, :-1].values, dic)
    seen_y = pd.concat([df0.iloc[:20000], df1.iloc[:n_sup], df2.iloc[:n_sup], df3.iloc[:n_sup], df4.iloc[:n_sup]],
                       axis=0, ignore_index=True).iloc[:, -1].values

    unseen_x = key2num_DeepSVDD(df0.iloc[20000:40000].iloc[:, :-1].values, dic)

    test_x = key2num_DeepSVDD(
        pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0,
                  ignore_index=True).iloc[:, :-1].values, dic)
    test_y = pd.concat([df0.iloc[-2000:], df1.iloc[-27:], df2.iloc[-201:], df3.iloc[-10:], df4.iloc[-21:]], axis=0,
                       ignore_index=True).iloc[:, -1].values
    seen_x.extend(unseen_x)
    X = seen_x
    seen_y = np.where(seen_y > 0, 1, seen_y)
    y = np.concatenate([seen_y, [0 for _ in range(len(unseen_x))]])
    test_y = np.where(test_y > 0, 1, test_y)
    return X, y, test_x, test_y