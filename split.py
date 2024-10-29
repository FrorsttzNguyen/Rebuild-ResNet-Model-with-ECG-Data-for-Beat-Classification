import glob

from sklearn.model_selection import train_test_split

import wfdb
import numpy as np
import pandas as pd

if __name__ == '__main__':
    length = 320
    list_label = np.array(['N', 'L', 'R', 'V', 'A'])
    list_file_name = glob.glob('/usr/diem/Documents/data/mit/*.hea')
    X = np.zeros((0, 4))
    y = np.zeros((0, 1))
    for i in range(0, len(list_file_name)):
        filename = list_file_name[i]
        hea = wfdb.rdheader(filename[:-4])
        if 'MLII' not in hea.sig_name:
            continue
        ann = wfdb.rdann(filename[:-4], extension='atr')

        symbol = np.array(ann.symbol)
        sample = np.array(ann.sample)
        chose_index = np.isin(symbol, list_label) & (sample >= length / 2) & (sample < hea.sig_len - length / 2)
        sample = sample[chose_index]
        symbol = symbol[chose_index]

        start = sample - int(length / 2)
        end = sample + int(length / 2)
        channel = np.ones(len(sample)) * np.where(np.array(hea.sig_name) == 'MLII')[0][0]

        X = np.append(X, np.stack([np.repeat(filename[:-4], len(sample)),
                                   start,
                                   end,
                                   channel], axis=1), axis=0)
        y = np.append(y, symbol[:, None], axis=0)

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=42, test_size=0.3)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, stratify=y_val, random_state=42, test_size=0.5)

    df_train = pd.DataFrame(np.append(X_train, y_train, axis=1),
                            columns=['filename_lr', 'start', 'end', 'channel', 'Label'])
    df_val = pd.DataFrame(np.append(X_val, y_val, axis=1),
                          columns=['filename_lr', 'start', 'end', 'channel', 'Label'])
    df_test = pd.DataFrame(np.append(X_test, y_test, axis=1),
                           columns=['filename_lr', 'start', 'end', 'channel', 'Label'])

    df_train.to_csv('/usr/diem/Documents/code/mit_beat_classification/train_test/train.csv', index=False, header=True)
    df_val.to_csv('/usr/diem/Documents/code/mit_beat_classification/train_test/val.csv', index=False, header=True)
    df_test.to_csv('/usr/diem/Documents/code/mit_beat_classification/train_test/test.csv', index=False, header=True)
