from typing import Dict, List, Any
from numpy.linalg import inv
from datamodel import *
import numpy as np
import math


class KalmanFilter:
    def __init__(self, x0, Q, R, calc_error=False):
        self.xm = None
        self.xp = x0
        self.Pm = None
        self.Pp = np.zeros((x0.shape[0], x0.shape[0]))
        self.Q = Q
        self.R = R
        self.I = np.eye(x0.shape[0])
        self.gain = None

        self.calc_error = calc_error
        self.eta = []
        self.epsilon = []

    def __call__(self, y1, y2):
        y2 = np.array([[y2, 1]])
        self.xm = self.xp
        self.PM = self.Pp + self.Q
        MV = y2 @ self.PM @ y2.T + self.R
        gain = (self.PM @ y2.T) / MV
        self.xp = self.xm + gain @ (y1 - y2 @ self.xm)
        z = (self.I - gain @ y2)
        self.PP = z @ self.PM @ z.T + gain @ gain.T * self.R

        if self.calc_error:
            self.eta.append(gain @ (y1 - y2 @ self.xm).flatten())
            self.epsilon.append((y1 - y2 @ self.xm).flatten())

        return list(self.xp.flatten()) + list(MV.flatten())


class Logger:
    local: True
    local_logs: dict[int, str] = {}

    def __init__(self, local=False) -> None:
        self.logs = ""
        self.local = local

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        output = json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True)
        if self.local:
            self.local_logs[state.timestamp] = output
        print(output)

        self.logs = ""


class BaseStrategy:
    def __init__(self):
        self.orders = {}
        self.state = None
        self.current_steps = 0
        self.data = {}

    def accumulate(self):
        pass

    def strategy(self):
        raise NotImplementedError

    def __call__(self, state: TradingState) -> Dict[str, List[Order]]:
        self.state = state
        self.orders = {
            'PEARLS': [],
            'BANANAS': [],
            'COCONUTS': [],
            'PINA_COLADAS': [],
        }
        self.accumulate()
        self.current_steps += 1
        self.strategy()
        return self.orders


class AvellanedaMM(BaseStrategy):
    def __init__(self, product: str, y: float, k: float, limit: int = 20, vol_window: int = 30):
        super().__init__()
        self.product = product
        self.y = y
        self.k = k
        self.limit = limit
        self.vol_window = vol_window

        self.data = {
            'bid': [],
            'ask': [],
            'mid_price': [],
            'log_return': [],
        }

    def accumulate(self):
        if self.product in self.state.order_depths.keys():
            depth = self.state.order_depths[self.product]
            self.data['bid'].append(min(depth.buy_orders.keys()))
            self.data['ask'].append(max(depth.sell_orders.keys()))
            self.data['mid_price'].append(
                (self.data['bid'][-1] + self.data['ask'][-1]) / 2)

            if self.current_steps > 1:
                self.data['log_return'].append(
                    math.log(self.data['mid_price'][-1] / self.data['mid_price'][-2]))

    def strategy(self):
        if self.current_steps < self.vol_window + 1:
            return

        vol = np.std(self.data['log_return'][-self.vol_window:]) ** 2
        s = self.data['mid_price'][-1]
        q = self.state.position.get(self.product, 0)
        r = s - q * self.y * vol
        spread = self.y * vol + (2 / self.y) * math.log(1 + self.y / self.k)
        bid = r - spread / 2
        ask = r + spread / 2
        bid_amount = self.limit - q
        ask_amount = -self.limit - q

        if bid_amount > 0:
            self.orders[self.product].append(
                Order(self.product, bid, bid_amount))

        if ask_amount < 0:
            self.orders[self.product].append(
                Order(self.product, ask, ask_amount))


class BolBStrategy(BaseStrategy):
    def __init__(self, product, smoothing: int = 20, stds: int = 2, limit=20):
        super().__init__()
        self.smoothing = smoothing
        self.stds = stds
        self.limit = limit

        self.product = product
        self.data = {'mid_price': []}

    def accumulate(self):
        if self.product in self.state.order_depths.keys():
            depth = self.state.order_depths[self.product]
            self.data['top_bid'] = max(depth.buy_orders.keys())
            self.data['top_ask'] = min(depth.sell_orders.keys())
            self.data['mid_price'].append(
                (self.data['top_bid'] + self.data['top_ask']) / 2)

    def strategy(self):
        if self.current_steps < self.smoothing:
            return

        prices = self.data['mid_price'][self.current_steps - self.smoothing:]
        bolu = np.mean(prices) + self.stds * np.std(prices)
        bold = np.mean(prices) - self.stds * np.std(prices)

        q = self.state.position.get(self.product, 0)
        bid_amount = self.limit - q
        ask_amount = -self.limit - q

        if prices[-1] > bolu and ask_amount < 0:
            self.orders[self.product] = [
                Order(self.product, self.data['top_bid'], ask_amount)]

        elif prices[-1] < bold and bid_amount > 0:
            self.orders[self.product] = [
                Order(self.product, self.data['top_ask'], bid_amount)]


class GreatWall(BaseStrategy):
    def __init__(self, product, upper, lower, limit=20):
        super().__init__()
        self.product = product
        self.limit = limit
        self.upper = upper + 10000
        self.lower = lower + 10000

    def strategy(self):
        q = self.state.position.get(self.product, 0)
        bid_amount = self.limit - q
        ask_amount = -self.limit - q

        if bid_amount > 0:
            self.orders[self.product].append(Order(self.product, self.lower, bid_amount))

        if ask_amount < 0:
            self.orders[self.product].append(Order(self.product, self.upper, ask_amount))


class StatArb(BaseStrategy):
    def __init__(self, gamma, mu, thresh, limit=(600, 300)):
        super().__init__()
        self.gamma = gamma
        self.mu = mu
        self.limit = limit

        self.U = thresh
        self.L = -thresh

        self.products = ('COCONUTS', 'PINA_COLADAS')
        self.target_pos = (gamma * limit[1], limit[1])
        self.data = {
            'mid': [0.0, 0.0],
            'bid': [0.0, 0.0],
            'ask': [0.0, 0.0],
            'signal': 0
        }

    def place_order(self, i, target_pos):
        product = self.products[i]
        pos = self.state.position.get(product, 0)

        if pos == target_pos:
            return

        else:
            order_size = target_pos - pos

        if order_size > 0:
            price = self.data['ask'][i]

        elif order_size < 0:
            price = self.data['bid'][i]

        self.orders[product].append(Order(product, price, order_size))

    def calc_prices(self):
        for i, product in enumerate(self.products):
            depth1 = self.state.order_depths[product]
            self.data['bid'][i] = min(depth1.buy_orders.keys())
            self.data['ask'][i] = max(depth1.sell_orders.keys())
            tb = max(depth1.buy_orders.keys())
            ta = min(depth1.sell_orders.keys())
            self.data['mid'][i] = (tb + ta) / 2

    def accumulate(self):
        self.calc_prices()
        self.data['signal'] = self.data['mid'][1] - self.gamma * self.data['mid'][0] - self.mu

    def strategy(self):
        signal = self.data['signal']

        if signal is None:
            return

        if signal > self.U:
            for i in range(2):
                target = (self.target_pos[0], -self.target_pos[1])[i]
                self.place_order(i, target)

        elif signal < self.L:
            for i in range(2):
                target = (self.target_pos[0], -self.target_pos[1])[i]
                self.place_order(i, -target)

        elif 0.1 * self.L < signal < 0.1 * self.U:
            for i in range(2):
                self.place_order(i, 0)


class RollLS(StatArb):
    def __init__(self, window, thresh, limit=(600, 300)):
        super().__init__(0, 0, thresh, limit)
        self.window = window
        self.out = None
        self.data = {
            'mid': [[], []],
            'bid': [0.0, 0.0],
            'ask': [0.0, 0.0],
            'signal': 0
        }

    def accumulate(self):
        self.calc_prices()

        if self.current_steps < self.window:
            self.data['signal'] = None
            return

        X = np.vstack((np.asarray(self.data['mid'][0][-self.window:]),
                       np.ones(self.window))).reshape(-1, 2)
        Y = np.asarray(self.data['mid'][1][-self.window:]).reshape(-1, 1)
        self.out = np.linalg.lstsq(X, Y, rcond=None)[0].flatten()
        self.data['signal'] = self.data['mid'][1][-1] - self.out[0] * self.data['mid'][0][-1] - self.out[1]
        print( '{}'.format(self.data['signal']))
        self.target_pos = (self.out[0] * self.limit[1], self.limit[1])


class Kalman(StatArb):
    def __init__(self, init_gamma, init_mu, q, r, thresh, window, limit=(600, 300)):
        super().__init__(init_gamma, init_mu, thresh, limit)
        self.out = (init_gamma, init_mu)
        self.target_pos = (self.out[0] * self.limit[1], self.limit[1])
        init_params = np.array([init_gamma, init_mu]).reshape(2, 1)
        self.window = window
        self.kf = KalmanFilter(init_params, q, r)

    def accumulate(self):
        self.calc_prices()

        if self.current_steps % self.window == 0:
            self.out = self.kf(self.data['mid'][1], self.data['mid'][0])

        self.data['signal'] = self.data['mid'][1] - self.out[0] * self.data['mid'][0] - self.out[1]

        '''self.target_pos = (self.out[0] * self.limit[1], self.limit[1])

        if self.target_pos[0] > self.limit[0]:
            self.target_pos = (self.limit[0], self.limit[1])'''


class Trader:
    def __init__(self, local=False):
        Q = np.asarray([[4.58648333e-08, 0],
                        [0, 7.08011170e-16]])
        self.strategies = [  # GreatWall('PEARLS', 1.99, -1.99),
            # AvellanedaMM('BANANAS', 5, 0.01),
            StatArb(1.549153, 2615.272303, 40)
        ]
        self.logger = Logger(local=local)

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for strategy in self.strategies:
            strategy_out = strategy(state)

            for product, orders in strategy_out.items():
                result[product] = result.get(product, []) + orders

        self.logger.flush(state, result)
        return result
