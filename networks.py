from torch.utils.data import Dataset, DataLoader
from utils import *
import torch.nn.functional as nnf
import lightning.pytorch as pl
import torch.nn as nn
import pandas as pd
import numpy as np
import torch


def av_return_loss(weights, returns):
    return torch.mean(weights * returns) * 100


class Unsupervised(Dataset):
    def __init__(self, data, window, features, forward_length):
        self.data = data.loc[:, features].to_numpy().astype(np.float32)
        self.window = window
        self.features = data.loc[:, 'r'].to_numpy().astype(np.float32)
        self.forward_length = forward_length

    def __len__(self):
        return len(self.data) - self.window - self.forward_length - 1

    def __getitem__(self, index):
        index += 1
        x = self.data[index:index + self.window]
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        x = x.flatten()
        y = self.features[index + self.window: index + self.window + self.forward_length].flatten()
        return torch.from_numpy(x), torch.from_numpy(y)


class GRUTrader(pl.LightningModule):
    def __init__(self, num_features, window, forward_length, num_layers, hidden_dim=100, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.dropout = dropout

        self.norm = nn.InstanceNorm1d(num_features * window)
        self.gru = nn.GRU(num_features * window, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.to_out = nn.Linear(hidden_dim, forward_length)

        self.h = torch.zeros(num_layers, hidden_dim).requires_grad_()

    def forward(self, x):
        x, self.h = self.gru(x, self.h.detach())
        x = self.to_out(x)
        x = nnf.dropout(x, p=self.dropout)
        return nnf.tanh(x)

    def training_step(self, batch, batch_nb):
        x, r = batch
        y = self(x)
        loss = av_return_loss(y, r)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, r = batch
        y = self(x)
        loss = av_return_loss(y, r)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, maximize=True)


if __name__ == '__main__':
    reader = PriceReader()
    trn = reader(['BERRIES'], [0, 1])
    val = reader(['BERRIES'], [2])
    features = ['r', 'perc_time']
    target = ['r']
    window = 32
    forward_length = 2

    train_set = Unsupervised(trn, window, features, forward_length)
    val_set = Unsupervised(val, window, features, forward_length)
    tl = DataLoader(train_set, batch_size=32, num_workers=2, shuffle=False)
    vl = DataLoader(val_set, batch_size=32, num_workers=2, shuffle=False)

    model = GRUTrader(num_features=len(features),
                      window=window,
                      forward_length=forward_length,
                      num_layers=1,
                      hidden_dim=100,
                      dropout=0.1)

    trainer = pl.Trainer(
        accelerator='cpu',
        max_epochs=50,
        callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
        fast_dev_run=False
    )

    trainer.fit(model, tl, vl)
