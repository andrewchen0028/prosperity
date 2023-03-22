import pandas as pd
import torch
import torch.nn
from torch.utils.data import Dataset
import lightning as pl

'''
DSBanana(Dataset):
  def __init__(self):
    ob = pd.read_csv("round1/ob_bnn_train.csv", delimiter=";", index_col="timestamp")


Model(pl.Lightning)
'''