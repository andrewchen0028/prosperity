import pandas as pd


def load_product(obj, product):
    df = pd.read_csv(f'round2/ob_{obj}.csv', delimiter=';', index_col=0)
    return df.loc[df['product'] == product]