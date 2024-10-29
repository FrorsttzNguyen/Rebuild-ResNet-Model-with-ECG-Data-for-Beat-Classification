import copy
import csv
import os
import time

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import torch
import tqdm
import json
from util20 import get_loaders
from eval import evaluate_for_confusion
from ecgresnet import ECGResNet

params = json.load(open('config_PKT.json', 'r'))
freq = 10
factor = 0.1
verbose = 1
default_lr = params['learning_rate']


def train():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    device = 'cuda'

    print('Device:', torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu')

    # adjuster = ADJUSTLR(freq, factor, verbose, default_lr)
    # train_loader, val_loader, _ = get_data_loader()

    train_loader, val_loader, test_loader = get_loaders(params, True)
    # model = DACLSTM(n_classes=N_CLASSES).to(device)

    model = ECGResNet(in_length=params['in_length'], in_channels=params['in_channels'], n_grps=params['n_grps'],
                      N=params['N'], num_classes=params['num_classes'], dropout=params['dropout'],
                      first_width=params['first_width'], stride=params['stride'], dilation=params['dilation']).to(
        device)
    # best_checkpoint_path = './logs/training_fold_test_batch/checkpoint_18_k0.pth'
    # checkpoint = torch.load(best_checkpoint_path)
    #
    # checkpoint['model_state_dict']['mlp_head.weight'] = checkpoint['model_state_dict']['mlp_head.weight'][:5]
    # checkpoint['model_state_dict']['mlp_head.bias'] = checkpoint['model_state_dict']['mlp_head.bias'][:5]

    # model.load_state_dict(checkpoint['model_state_dict'])

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETER_CONFIG['lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate']
                                 # lr=adjuster.get_new_lr()
                                 )

    criterion = torch.nn.CrossEntropyLoss()

    EPOCHS = params['epochs']

    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    print('Start training...')
    print('Epochs:', EPOCHS)
    print('Iterations per training epoch:', len(train_loader))
    print('Iterations per validation epoch:', len(test_loader))
    # print('Iterations per validation epoch:', len(val_loader))
    max_f1 = 0
    avg_train_loss = []
    avg_val_loss = []
    with open('./results/log_val_cf.csv', 'w') as f:
        pass
    for epoch in range(EPOCHS):

        print(f'------------- Epoch [{epoch + 1}/{EPOCHS}] -------------')
        t0 = time.time()

        # Training
        train_loss = 0
        # adjuster.on_epoch_end(epoch)
        model.train()
        # continue
        for i, samples in enumerate(tqdm.tqdm(train_loader)):
            # get data
            # x = samples['waveform']
            tensorData, _, labels, _ = samples
            # if i == 0:
            #     print(labels[0])
            bs, c, h, w = tensorData.shape
            x = tensorData.reshape(bs, c, h * w)
            # print("x: ", x)
            # labels = samples['label']

            # clear gradient
            optimizer.zero_grad()
            # forward pass
            logits = model(x.to(device))
            # print(logits.shape, logits, labels.shape, labels)
            # calculate loss and metric
            loss = criterion(logits.to(device).float(), labels.to(device).float())
            # logits_ = logits.detach()
            # logits_[logits_ >= 0.5] = 1
            # logits_[logits_ < 0.5] = 0

            # backprop and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_loss = 0
        with torch.no_grad():
            confusion_matrix = np.zeros((4, params['num_classes']))
            model.eval()
            # for i, samples in enumerate(tqdm.tqdm(val_loader)):
            for i, samples in enumerate(tqdm.tqdm(val_loader)):
                # get data
                # x = samples['waveform']
                tensorData, _, labels, _ = samples
                # print(tensorData.shape)
                bs, c, h, w = tensorData.shape
                x = tensorData.reshape(bs, c, h * w)
                # print("x: ", x)
                # labels = samples['label']

                # clear gradient
                optimizer.zero_grad()
                # forward pass
                logits = model(x.to(device))
                # print(logits.shape, logits, labels.shape, labels)
                # calculate loss and metric
                loss = criterion(logits.to(device).float(), labels.to(device).float())
                logits_ = logits.detach()
                logits_[logits_ >= 0.5] = 1
                logits_[logits_ < 0.5] = 0

                val_loss += loss.item()

                # sub_confusion_matrix = evaluate_for_confusion(np.eye(N_CLASSES)[labels],
                #                                               np.eye(N_CLASSES)[logits.argmax(1)])
                sub_confusion_matrix = evaluate_for_confusion(np.array(labels.cpu()),
                                                              np.array(logits_.cpu()),
                                                              num_classes=params['num_classes'])
                confusion_matrix += sub_confusion_matrix

        # with open('./logs/csv_test/log_val_cf.csv', 'a') as f
        with open('./results/log_val_cf.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(np.array(['Epochs', epoch]))
            event_types = np.array(
                 ['N', 'L', 'R', 'V', 'A'])
            event_types = np.array(
                [''] + list(params['dict_label'].keys()))
            writer.writerow(event_types)
            for i in range(4):
                if i == 0:
                    row = np.insert(confusion_matrix[i].astype(str), 0, 'TP')
                elif i == 1:
                    row = np.insert(confusion_matrix[i].astype(str), 0, 'FN')
                elif i == 2:
                    row = np.insert(confusion_matrix[i].astype(str), 0, 'FP')
                else:
                    row = np.insert(confusion_matrix[i].astype(str), 0, 'TN')
                writer.writerow(row)
            SE = np.nan_to_num(
                confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[1]))
            P = np.nan_to_num(
                confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[2]))
            F1 = np.nan_to_num(2 * SE * P / (SE + P))
            SE = np.append(SE, np.mean(SE))
            P = np.append(P, np.mean(P))
            F1 = np.append(F1, np.mean(F1))
            writer.writerow(np.insert(SE.astype(str), 0, 'SE'))
            writer.writerow(np.insert(P.astype(str), 0, 'P+'))
            writer.writerow(np.insert(F1.astype(str), 0, 'F1'))

            writer.writerow(np.array(['']))
        f.close()

        avg_train_loss = avg_train_loss + [train_loss / len(train_loader)]
        avg_val_loss = avg_val_loss + [val_loss / len(test_loader)]
        print("Epoch[{}/{}] - Loss: {:.4f} - ValLoss: {:.4f} - ValAvgF1: {:.4f}, ETA: {:.0f}s"
              .format(epoch + 1, EPOCHS, avg_train_loss[-1], avg_val_loss[-1], F1[-1],
                      time.time() - t0))
        if max_f1 < F1[-1]:
            max_f1 = F1[-1]
            print("Max at epoch ", epoch)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': H,
            }, os.path.join(os.getcwd(), f'training/max_checkpoint.pth'))

        H["train_loss"].append(avg_train_loss)
        H["val_loss"].append(avg_val_loss)

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': H,
        }, os.path.join(os.getcwd(), f'training/checkpoint_{epoch + 1}.pth'))

    plt.figure()
    plt.plot(avg_train_loss)
    plt.plot(avg_val_loss)
    plt.show()


if __name__ == '__main__':
    train()
    # test()
