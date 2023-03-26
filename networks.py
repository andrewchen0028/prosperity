from torch.utils.data import Dataset, DataLoader
from utils import *
import torch.nn.functional as nnf
import lightning.pytorch as pl
import torch.nn as nn
import pandas as pd
import numpy as np
import torch


def av_return_loss(weights, returns, penalty=0.001):
    ret = torch.prod(weights * returns * 100, dim=-1)
    penalty = torch.diff(weights, dim=-1) * penalty
    ret -= torch.sum(penalty ** 2, dim=1)
    return torch.mean(ret)


class Unsupervised(Dataset):
    def __init__(self, data, window, features, forward_length):
        self.window = window
        self.data = data.loc[:, features].to_numpy().astype(np.float32)
        self.features = data.loc[:, 'r'].to_numpy().astype(np.float32)
        self.time = data.loc[:, 'perc_time'].to_numpy().astype(np.float32)
        self.forward_length = forward_length

    def __len__(self):
        return len(self.data) - self.window - self.forward_length

    def __getitem__(self, index):
        x = self.data[index:index + self.window]
        x = (x - x.mean(axis=0)) / x.std(axis=0)
        x = x.flatten() + self.time[index:index + self.window].flatten()
        y = self.features[index + self.window: index + self.window + self.forward_length].flatten()
        return torch.from_numpy(x), torch.from_numpy(y)


class GRU(nn.Module):
    def __init__(self, num_features, window, forward_length, num_layers, hidden_dim=100, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        self.norm = nn.InstanceNorm1d(num_features * window)
        self.gru = nn.GRU(num_features * window, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.to_out = nn.Linear(hidden_dim, forward_length)

        self.h = torch.zeros(num_layers, hidden_dim).cuda().requires_grad_()

    def forward(self, x):
        x, self.h = self.gru(x, self.h.detach())
        x = self.to_out(x)
        x = nnf.dropout(x, p=self.dropout)
        return nnf.tanh(x)


class MLP(nn.Module):
    def __init__(self, num_features, window, forward_length, num_layers, hidden_dim=50, dropout=0.1):
        super().__init__()
        self.dropout = dropout

        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())

        self.to_hidden = nn.Linear(num_features * window, hidden_dim)
        self.hidden = nn.Sequential(*layers)
        self.to_out = nn.Linear(hidden_dim, forward_length)

    def forward(self, x):
        x = self.to_hidden(x)
        x = self.hidden(x)
        x = self.to_out(x)
        return nnf.tanh(x)


class NetTrader(pl.LightningModule):
    def __init__(self, model_type, **model_params):
        super().__init__()
        self.save_hyperparameters()

        if model_type == 'gru':
            self.model = GRU(**model_params)

        elif model_type == 'mlp':
            self.model = MLP(**model_params)

    def forward(self, x):
        return self.model(x)

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
    torch.set_float32_matmul_precision('high')

    reader = PriceReader()
    trn = reader(['BERRIES'], [0, 1])
    val = reader(['BERRIES'], [2])
    features = ['r']
    target = ['r']
    window = 24
    forward_length = 10

    train_set = Unsupervised(trn, window, features, forward_length)
    val_set = Unsupervised(val, window, features, forward_length)
    tl = DataLoader(train_set, batch_size=32, num_workers=16, shuffle=False)
    vl = DataLoader(val_set, batch_size=32, num_workers=16, shuffle=False)

    model = NetTrader(
        'mlp',
        num_features=len(features),
        window=window,
        forward_length=forward_length,
        num_layers=1,
        hidden_dim=40,
        dropout=0.1
    )

    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        mode='max',
        save_top_k=1,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=50,
        callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='max'),
                   checkpoint],
        fast_dev_run=False
    )

    trainer.fit(model, tl, vl)
