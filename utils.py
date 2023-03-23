import pandas as pd


def load_product(obj, product):
    df = pd.read_csv(f'round2/ob_{obj}.csv', delimiter=';')
    return df.loc[df['product'] == product]