import os

from matplotlib import pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

import wfdb

from preprocessing import butter_bandpass_filter
from transforms import ToTensor
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from dataset import collate_fn


class RealDataset(Dataset):
    def __init__(self):
        self.topic = topic
        self.dataset = dataset
        self.waveform_dir = waveform_dir
        self.in_length = in_length
        self.length = length
        self.transform = transform
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dict_label = dict_label
        self.fs = fs

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx == 0:
            self.dataset = self.dataset.sample(frac=1)

        waveform, target, sample_id = self.get_waveform(idx)

        sample = {'waveform': waveform, 'id': sample_id, 'label': target}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_waveform(self, idx):
        raw_signal, fields = wfdb.rdsamp(os.path.join(self.waveform_dir, self.dataset['filename_lr'][idx]))
        raw_signal = raw_signal[self.dataset['start'][idx]:self.dataset['end'][idx], int(self.dataset['channel'][idx])]
        waveform = np.array(raw_signal.T)
        waveform = np.nan_to_num(waveform)
        label = self.dataset['Label'][idx].split(',')
        target = np.zeros(self.num_classes)
        for i in range(0, len(label)):
            target[self.dict_label[label[i]]] = 1

        return waveform, target, idx


def get_loaders(params, topic):
    transform = transforms.Compose([ToTensor()])

    train_df = pd.read_csv(params['train_labels_csv'])
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(params['val_labels_csv'])
    val_df = val_df.sample(frac=1).reset_index(drop=True)
    testset_df = pd.read_csv(params['test_labels_csv'])

    trainset = RealDataset(len(train_df), topic, params['data_dir'], params['in_length'], train_df,
                           transform=transform, in_channels=params['in_channels'], num_classes=params['num_classes'],
                           dict_label=params['dict_label'], fs=params['fs'])

    valset = RealDataset(len(val_df), topic, params['data_dir'], params['in_length'],
                         val_df, transform=transform, in_channels=params['in_channels'],
                         num_classes=params['num_classes'], fs=params['fs'],
                         dict_label=params['dict_label'])

    testset = RealDataset(len(testset_df), topic, params['data_dir'], params['in_length'],
                          testset_df, transform=transform, in_channels=params['in_channels'],
                          num_classes=params['num_classes'], fs=params['fs'],
                          dict_label=params['dict_label'])

    train_loader = DataLoader(trainset, batch_size=params['batch_size'],
                              num_workers=4, collate_fn=collate_fn, shuffle=True)

    val_loader = DataLoader(valset, batch_size=params['batch_size'],
                            num_workers=4, collate_fn=collate_fn)

    test_loader = DataLoader(testset, batch_size=params['batch_size'],
                             num_workers=4, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader


