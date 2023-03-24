import pandas as pd


def load_product(obj, product, round):
    df = pd.read_csv(f'round{round}/ob_{obj}.csv', delimiter=';', index_col=0)
    return df.loc[df['product'] == product]


def load_trades(obj, product, round):
    df = pd.read_csv(f'round{round}/trd_{obj}.csv', delimiter=';', index_col=0)
    return df.loc[df['symbol'] == product]