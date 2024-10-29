import pandas as pd
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv('/usr/diem/Documents/data/ptb_xl_1.0.1/test.csv')
    # for i in range(0, len(df)):
    #     count = df['filename_lr'].str.contains(df['filename_lr'][i])
    #     if np.sum(count) > 1:
    #         print(df['filename_lr'][i], 'at', np.where(count))

    classes = ["NORM", "CD", "HYP", "MI", "STTC"]
    for i in classes:
        count = df['Label'].str.contains(i)
        print(i, ": ", np.sum(count))
    print("Total:", len(df))