import random
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import warnings


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


def preprocessing_UNSW(options, ratio=10, ratio_ab=1, n_class=5):
    dir = options['dataset_dir']
    n_sup = options['validation_size']
    seed = options['random_seed']
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
    # lst = [df0, df1, df4, df5, df6, df7]
    # print(f'samples for each classes: {[len(i) for i in lst]}')
    seen_x = df0.iloc[:100, :-1].values
    seen_y = df0.iloc[:100, -1].values
    if n_class == 5:
        sup_x = pd.concat([df1.iloc[:n_sup], df4.iloc[:n_sup], df5.iloc[:n_sup], df6.iloc[:n_sup], df7.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values
        sup_y = pd.concat([df1.iloc[:n_sup], df4.iloc[:n_sup], df5.iloc[:n_sup], df6.iloc[:n_sup], df7.iloc[:n_sup]], axis=0, ignore_index=True).iloc[:, -1].values
        unseen_x = pd.concat([df0.iloc[20000:int(20000+ratio*3000)], df1.iloc[n_sup:int(ratio_ab*3000)+n_sup], df4.iloc[n_sup:int(ratio_ab*3000)+n_sup], df5.iloc[n_sup:int(ratio_ab*3000)+n_sup], df6.iloc[n_sup:int(ratio_ab*3000)+n_sup], df7.iloc[n_sup:int(ratio_ab*3000)+n_sup]], axis=0, ignore_index=True).iloc[:, :-1].values
        unseen_y = pd.concat([df0.iloc[20000:int(20000+ratio*3000)], df1.iloc[n_sup:int(ratio_ab*3000)+n_sup], df4.iloc[n_sup:int(ratio_ab*3000)+n_sup], df5.iloc[n_sup:int(ratio_ab*3000)+n_sup], df6.iloc[n_sup:int(ratio_ab*3000)+n_sup], df7.iloc[n_sup:int(ratio_ab*3000)+n_sup]], axis=0, ignore_index=True).iloc[:, -1].values
        test_x = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:], df6.iloc[-2000:], df7.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, :-1].values
        test_y = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:], df6.iloc[-2000:], df7.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, -1].values
    elif n_class == 4:
        sup_x = pd.concat([df1.iloc[:n_sup], df4.iloc[:n_sup], df5.iloc[:n_sup], df6.iloc[:n_sup]],
                          axis=0, ignore_index=True).iloc[:, :-1].values
        sup_y = pd.concat([df1.iloc[:n_sup], df4.iloc[:n_sup], df5.iloc[:n_sup], df6.iloc[:n_sup]],
                          axis=0, ignore_index=True).iloc[:, -1].values
        unseen_x = pd.concat(
            [df0.iloc[20000:int(20000 + ratio * 3000)], df1.iloc[n_sup:3000 + n_sup], df4.iloc[n_sup:3000 + n_sup],
             df5.iloc[n_sup:3000 + n_sup], df6.iloc[n_sup:3000 + n_sup]], axis=0,
            ignore_index=True).iloc[:, :-1].values
        unseen_y = pd.concat(
            [df0.iloc[20000:int(20000 + ratio * 3000)], df1.iloc[n_sup:3000 + n_sup], df4.iloc[n_sup:3000 + n_sup],
             df5.iloc[n_sup:3000 + n_sup], df6.iloc[n_sup:3000 + n_sup]], axis=0,
            ignore_index=True).iloc[:, -1].values
        test_x = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:], df6.iloc[-2000:]
                            ], axis=0, ignore_index=True).iloc[:, :-1].values
        test_y = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:], df6.iloc[-2000:]
                            ], axis=0, ignore_index=True).iloc[:, -1].values
    elif n_class == 3:
        sup_x = pd.concat([df1.iloc[:n_sup], df4.iloc[:n_sup], df5.iloc[:n_sup]],
                          axis=0, ignore_index=True).iloc[:, :-1].values
        sup_y = pd.concat([df1.iloc[:n_sup], df4.iloc[:n_sup], df5.iloc[:n_sup]],
                          axis=0, ignore_index=True).iloc[:, -1].values
        unseen_x = pd.concat(
            [df0.iloc[20000:int(20000 + ratio * 3000)], df1.iloc[n_sup:3000 + n_sup], df4.iloc[n_sup:3000 + n_sup],
             df5.iloc[n_sup:3000 + n_sup]], axis=0,
            ignore_index=True).iloc[:, :-1].values
        unseen_y = pd.concat(
            [df0.iloc[20000:int(20000 + ratio * 3000)], df1.iloc[n_sup:3000 + n_sup], df4.iloc[n_sup:3000 + n_sup],
             df5.iloc[n_sup:3000 + n_sup]], axis=0,
            ignore_index=True).iloc[:, -1].values
        test_x = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, :-1].values
        test_y = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:], df5.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, -1].values
    elif n_class == 2:
        sup_x = pd.concat([df1.iloc[:n_sup], df4.iloc[:n_sup]],
                          axis=0, ignore_index=True).iloc[:, :-1].values
        sup_y = pd.concat([df1.iloc[:n_sup], df4.iloc[:n_sup]],
                          axis=0, ignore_index=True).iloc[:, -1].values
        unseen_x = pd.concat(
            [df0.iloc[20000:int(20000 + ratio * 3000)], df1.iloc[n_sup:3000 + n_sup], df4.iloc[n_sup:3000 + n_sup]], axis=0,
            ignore_index=True).iloc[:, :-1].values
        unseen_y = pd.concat(
            [df0.iloc[20000:int(20000 + ratio * 3000)], df1.iloc[n_sup:3000 + n_sup], df4.iloc[n_sup:3000 + n_sup]], axis=0,
            ignore_index=True).iloc[:, -1].values
        test_x = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, :-1].values
        test_y = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:], df4.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, -1].values
    elif n_class == 1:
        sup_x = df1.iloc[:n_sup].iloc[:, :-1].values
        sup_y = df1.iloc[:n_sup].iloc[:, -1].values
        unseen_x = pd.concat(
            [df0.iloc[20000:int(20000 + ratio * 3000)], df1.iloc[n_sup:3000 + n_sup]], axis=0,
            ignore_index=True).iloc[:, :-1].values
        unseen_y = pd.concat(
            [df0.iloc[20000:int(20000 + ratio * 3000)], df1.iloc[n_sup:3000 + n_sup]], axis=0,
            ignore_index=True).iloc[:, -1].values
        test_x = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, :-1].values
        test_y = pd.concat([df0.iloc[-20000:], df1.iloc[-2000:]], axis=0, ignore_index=True).iloc[:, -1].values

    return seen_x, seen_y, sup_x, sup_y, unseen_x, unseen_y, test_x, test_y



def data2df(x, y):
    df = pd.DataFrame(x)
    df['y_true'] = y
    df['y_pred'] = y
    df['dist'] = 0
    return df

def getfpr95tpr(y_true, dist, steps=10000):
    start = min(dist)
    end = max(dist)
    gap = (end - start) / steps
    count = 0.0
    fpr_all = 0.0
    for delta in np.arange(start, end, gap):
        y_pred = np.zeros(len(y_true))
        y_pred = np.where(dist >= delta, 1, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        if tpr <= 0.96 and tpr >= 0.94:
            fpr_all += fpr
            count += 1
    if count == 0:
        for delta in np.arange(start, end, gap):
            y_pred = np.zeros(len(y_true))
            y_pred = np.where(dist >= delta, 1, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
            if tpr >= 0.94:
                fpr_all += fpr
                count += 1
    if count == 0:
        fpr_final = 1
    else:
        fpr_final = fpr_all / count
    return fpr_final