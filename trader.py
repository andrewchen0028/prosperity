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
        gain = self.PM @ y2.T @ inv(y2 @ self.PM @ y2.T + self.R)
        self.xp = self.xm + gain @ (y1 - y2 @ self.xm)
        z = (self.I - gain @ y2)
        self.PP = z @ self.PM @ z.T + gain @ gain.T * self.R

        if self.calc_error:
            self.eta.append(gain @ (y1 - y2 @ self.xm).flatten())
            self.epsilon.append((y1 - y2 @ self.xm).flatten())

        return self.xp.flatten()


class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        print(json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

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


class NaiveMM(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.product = "PEARLS"
        self.pearls_limit = 20

    def strategy(self) -> Dict[str, List[Order]]:
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """

        for product in self.state.order_depths.keys():
            if product == "PEARLS":
                # Mean reversion strategy
                bids: Dict[int, int] = self.state.order_depths[product].buy_orders
                asks: Dict[int, int] = self.state.order_depths[product].sell_orders
                print("bids: ", bids)
                print("asks: ", asks)
                position: int = self.state.position["PEARLS"] \
                    if "PEARLS" in self.state.position.keys() else 0
                orders: list[Order] = []

                # Get best bid, best ask, and mid price
                # NOTE: The if statements may not be necessary, not sure if
                #       this challenge involves no-bid or no-ask situations?
                if bids:
                    best_bid = max(bids.keys())
                    best_bid_volume = bids[best_bid]
                if asks:
                    best_ask = min(asks.keys())
                    best_ask_volume = asks[best_ask]

                # TODO: Make these two "if" statements handle the situation
                #           (best ask volume) < (remainder of position limit).
                #       I.e., fill the "second-best bid/ask".
                # NOTE: positive quantity => buy order

                # Fill asks below price
                if position < self.pearls_limit and best_ask < 10000:
                    qty = max(position - self.pearls_limit,
                              best_ask_volume)  # NEGATIVE
                    print("best_ask_volume: ", best_ask_volume)
                    print("BUY", str(-qty), "at", best_ask)
                    orders.append(Order(product, best_ask, -qty))

                    if position + abs(qty) > 0:
                        # We will be long pearls after this purchase.
                        # Ask just above fair value.
                        orders.append(
                            Order(product, 10000, -(position + abs(qty))))
                        print("ASK", str(-(position + abs(qty))), "at", 10000)
                    elif position + abs(qty) < 0:
                        # We will be short pearls after this purchase.
                        # Bid just below fair value.
                        orders.append(
                            Order(product, 10000, -(position + abs(qty))))
                        print("BID", str(-(position + abs(qty))), "at", 10000)

                # Fill bids above price
                if position > -self.pearls_limit and best_bid > 10000:
                    qty = min(position + self.pearls_limit,
                              best_bid_volume)  # POSITIVE
                    print("best_bid_volume: ", best_bid_volume)
                    print("SELL", str(qty), "at", best_bid)
                    orders.append(Order(product, best_bid, -qty))

                    if position - abs(qty) > 0:
                        # We will be long pearls after this sale.
                        # Ask just above fair value.
                        orders.append(
                            Order(product, 10000, -(position - abs(qty))))
                        print("ASK", str(-(position - abs(qty))), "at", 10000)
                    elif position - abs(qty) < 0:
                        # We will be short pearls after this sale.
                        # Bid just below fair value.
                        orders.append(
                            Order(product, 10000, -(position - abs(qty))))
                        print("BID", str(-(position - abs(qty))), "at", 10000)

                # Add orders to result dict
                self.orders[product] = orders


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
    def __init__(self, gamma, mu, u_thresh, l_thresh, exit_thresh, limit=(600, 300)):
        super().__init__()
        self.gamma = gamma
        self.mu = mu
        self.U = u_thresh
        self.L = l_thresh
        self.exit = exit_thresh
        self.limit = limit

        self.products = ('COCONUTS', 'PINA_COLADAS')
        self.target_pos = (limit[0], limit[0] / gamma)
        self.data = {
            'mid': [0.0, 0.0],
            'bid': [0.0, 0.0],
            'ask': [0.0, 0.0],
            'signal': 0
        }
        self.current_steps = 0

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

    def accumulate(self):
        for i, product in enumerate(self.products):
            depth1 = self.state.order_depths[product]
            self.data['bid'][i] = min(depth1.buy_orders.keys())
            self.data['ask'][i] = max(depth1.sell_orders.keys())
            tb = max(depth1.buy_orders.keys())
            ta = min(depth1.sell_orders.keys())
            self.data['mid'][i] = (tb + ta) / 2

        self.data['signal'] = self.data['mid'][1] - self.gamma * self.data['mid'][0] - self.mu

    def strategy(self):
        if self.current_steps == 0:
            return

        signal = self.data['signal']

        if signal > self.U:
            for i in range(2):
                target = (self.target_pos[0], -self.target_pos[1])[i]
                self.place_order(i, target)

        elif signal < self.L:
            for i in range(2):
                target = (self.target_pos[0], self.target_pos[1])[i]
                self.place_order(i, -target)

        elif -self.exit < signal < self.exit:
            for i in range(2):
                self.place_order(i, 0)


class Kalman(StatArb):
    def __init__(self, init_gamma, init_mu, q, r, u_thresh, l_thresh, exit_thresh, limit=(600, 300)):
        super().__init__(init_gamma, init_mu, u_thresh, l_thresh, exit_thresh, limit)
        init_params = np.array([init_gamma, init_mu]).reshape(2, 1)
        self.kf = KalmanFilter(init_params, q, r)

    def accumulate(self):
        for i, product in enumerate(self.products):
            depth1 = self.state.order_depths[product]
            self.data['bid'][i] = min(depth1.buy_orders.keys())
            self.data['ask'][i] = max(depth1.sell_orders.keys())
            tb = max(depth1.buy_orders.keys())
            ta = min(depth1.sell_orders.keys())
            self.data['mid'][i] = (tb + ta) / 2

        out = self.kf(self.data['mid'][1], self.data['mid'][0])
        self.data['signal'] = self.data['mid'][1] - out[0] * self.data['mid'][0] - out[1]

        self.target_pos = (self.limit[0], self.limit[0] / out[0])


class Trader:
    def __init__(self):
        Q = np.asarray([[0.00021424, 0],
                        [0, 0]])

        self.strategies = [GreatWall('PEARLS', 1.99, -1.99),
                           AvellanedaMM('BANANAS', 5, 0.01),
                           Kalman(1.927451, -439.161182, Q, 1.72464704, 0.0005, -0.0005, 0.0001)
                           ]

        self.logger = Logger()

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for strategy in self.strategies:
            strategy_out = strategy(state)

            for product, orders in strategy_out.items():
                result[product] = result.get(product, []) + orders

        self.logger.flush(state, result)
        return result

