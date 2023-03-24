from trader import *
from utils import *


INIT_P = np.array([[1.929786],
                   [-439.161149]])


def fit(Q, R):
    km = KalmanFilter(INIT_P, Q, R, calc_error=True)

    for row in train.to_numpy():
        km(row[1], row[0])

    return np.var(np.asarray(km.eta), axis=0), np.var(np.asarray(km.epsilon), axis=0)


if __name__ == '__main__':
    cnt = load_product('train', 'COCONUTS')
    pnc = load_product('train', 'PINA_COLADAS')
    train = pd.concat((cnt['mid_price'], pnc['mid_price']), axis=1)
    train.columns = ['cnt', 'pnc']

    res = fit(0.0001 * np.eye(2), 0.001)

    for i in range(50):
        print(res)
        res = fit(0.0001 * np.eye(2), res[1][0])
