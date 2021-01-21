import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import numpy as np

from dataset import *
from model import *
from utils import *


class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val


def train(model, device, train_loader, optimizer, scheduler, epoch, iter_meter):
    model.train()
    data_len = len(train_loader.dataset)
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels, input_lengths, label_lengths = _data 
        spectrograms, labels, input_lengths, label_lengths = \
            spectrograms.to(device), labels.to(device), input_lengths.to(device), label_lengths.to(device)

        optimizer.zero_grad()

        loss = model(spectrograms, input_lengths, labels, label_lengths)
        loss.backward()

        optimizer.step()
        scheduler.step()
        iter_meter.step()
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, epoch, iter_meter):
    print('\nevaluating...')
    model.eval()
    test_loss = 0
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, input_lengths = spectrograms.to(device), input_lengths.to(device)

            decoded_preds = model.recognize(spectrograms, input_lengths, device)
            decoded_targets = labels.tolist()

            for j in range(len(decoded_preds)):
                decoded_preds[j] = text_transform.int_to_text(decoded_preds[j])
                decoded_targets[j] = text_transform.int_to_text(decoded_targets[j])
                print(decoded_targets[j], decoded_preds[j])
                test_cer.append(cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(wer(decoded_targets[j], decoded_preds[j]))


    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    print('Test set: Average loss: {:.4f}, Average CER: {:4f} Average WER: {:.4f}\n'.format(test_loss, avg_cer, avg_wer))


def main(learning_rate=5e-4, batch_size=20, epochs=10,
        train_url="train-clean-100", test_url="test-clean", gpu=0):

    hparams= {
        "enc": {
            "type": "lstm",
            "hidden_size": 160,
            "output_size": 160,
            "n_layers": 2,
            "bidirectional": True,
        },
        "dec": {
            "type": "lstm",
            "hidden_size": 128,
            "output_size": 160,
            "n_layers": 1,
        },
        "joint":{
            "input_size": 320,
            "inner_size": 256,
        },
        "vocab_size": 29,
        "share_weight": False,
        "feature_dim": 64,
        "dropout": 0.3,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

    config = AttrDict(hparams)
    
    use_cuda =  torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda", gpu) if use_cuda else torch.device("cpu")

    if not os.path.isdir("./data"):
        os.makedirs("./data")

    train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)

    kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(x, 'train'),
                                **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(x, 'valid'),
                                **kwargs)

    model = Transducer(config).to(device)

    print(model)
    print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')
    
    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, scheduler, epoch, iter_meter)
        test(model, device, test_loader, epoch, iter_meter)


if __name__ == '__main__':
    learning_rate = 5e-4
    batch_size = 1
    epochs = 10
    libri_train_set = "train-clean-100"
    libri_test_set = "test-clean"
    gpu = 0

    main(learning_rate, batch_size, epochs, libri_train_set, libri_test_set, gpu)