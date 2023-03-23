from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as nnf
import lightning.pytorch as pl
import torch.nn as nn
import pandas as pd
import numpy as np
import torch


def load_product(obj, product):
    df = pd.read_csv(f'round2/ob_{obj}.csv', delimiter=';')
    return df.loc[df['product'] == product]


class DL(Dataset):
    def __init__(self, data, features, target, back_length, forward_length):
        self.data = data.loc[:, features].to_numpy().astype(np.float32)
        self.target = data.loc[:, target].to_numpy().astype(np.float32)
        self.back_length = back_length
        self.forward_length = forward_length

    def __len__(self):
        return len(self.data) - self.back_length - self.forward_length

    def __getitem__(self, index):
        x = self.data[index: index + self.back_length].flatten()
        y = self.target[index + self.back_length: index + self.back_length + self.forward_length].flatten()
        return torch.nan_to_num(torch.from_numpy(x)), torch.nan_to_num(torch.from_numpy(y))


class Net(pl.LightningModule):
    def __init__(self, num_features, back_length, forward_length, hidden_dim=100, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.lin1 = nn.Linear(num_features * back_length, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.lin3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.lin4 = nn.Linear(hidden_dim, forward_length)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(self.lin1(x))
        x = nnf.relu(x)
        x = self.dropout2(self.lin2(x))
        x = nnf.relu(x)
        x = self.dropout3(self.lin3(x))
        x = nnf.relu(x)
        x = self.dropout4(self.lin4(x))
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    bnn_trn = load_product('train', 'BANANAS')
    bnn_val = load_product('val', 'BANANAS')
    features = ['lr']
    target = ['lr']
    back_length = 20
    forward_length = 10

    trainloader = DataLoader(DL(bnn_trn, features, target, back_length, forward_length), batch_size=64, num_workers=2,
                             shuffle=True)

    valloader = DataLoader(DL(bnn_val, features, target, back_length, forward_length), batch_size=64, num_workers=2,
                           shuffle=False)

    model = Net(len(features), back_length, forward_length)

    trainer = pl.Trainer(
        accelerator='cpu',
        max_epochs=50,
        callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
        fast_dev_run=False
    )

    trainer.fit(model, trainloader, valloader)
