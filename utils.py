from io import BytesIO
import pandas as pd
import numpy as np


class PriceReader:
    def __init__(self):
        self.product_days = {
            'PEARLS': [-2, -1, 0, 1, 2],
            'BANANAS': [-2, -1, 0, 1, 2],
            'COCONUTS': [-1, 0, 1, 2],
            'PINA_COLADAS': [-1, 0, 1, 2],
            'DIVING_GEAR': [0, 1, 2],
            'BERRIES': [0, 1, 2],
            'DOLPHIN_SIGHTINGS': [0, 1, 2]
        }

    def read(self, products, day):
        for prod in products:
            if day not in self.product_days[prod]:
                raise ValueError(f'Product {prod} not available on day {day}')

        df = pd.read_csv('raw/prices_day_{}.csv'.format(day), delimiter=';', index_col=1)
        df = df.loc[df['product'].isin(products)]
        return df

    def calc_features(self, ret, length):
        ret['perc_time'] = list(np.linspace(0, 1, 10000)) * length
        ret['r'] = ret['mid_price'].pct_change()
        ret['r'] = ret['r'].fillna(0)
        return ret

    def __call__(self, products, days):
        days = sorted(days)

        for i, d in enumerate(days):
            if i == 0:
                ret = self.read(products, d)
                continue

            df = self.read(products, d)
            df.index += int(i * 1e6)
            ret = pd.concat((ret, df))

        return self.calc_features(ret, len(days))


def array_to_bytes(x: np.ndarray) -> bytes:
    np_bytes = BytesIO()
    np.save(np_bytes, x, allow_pickle=True)
    return np_bytes.getvalue()


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)
