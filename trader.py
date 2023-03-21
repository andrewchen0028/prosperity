from typing import Dict, List
from datamodel import *
import numpy as np
import math


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
        }
        self.accumulate()
        self.current_steps += 1
        self.strategy()
        return self.orders


class AvellanedaMM(BaseStrategy):
    def __init__(self, product: str, y: float, k: float, limit: int, max_t: float = 20000, vol_window: int = 30):
        super().__init__()
        self.product = product
        self.y = y
        self.k = 0
        self.ks = np.linspace(0.000000001, 10000000, 20000)
        self.limit = limit
        self.max_t = max_t
        self.vol_window = vol_window

        self.data = {
            'top_bid': [],
            'top_ask': [],
            'mid_price': [],
            'log_return': [],
        }

    def accumulate(self):
        if self.product in self.state.order_depths.keys():
            depth = self.state.order_depths[self.product]
            self.data['top_bid'].append(max(depth.buy_orders.keys()))
            self.data['top_ask'].append(min(depth.sell_orders.keys()))
            self.data['mid_price'].append((self.data['top_bid'][-1] + self.data['top_ask'][-1]) / 2)

            if self.current_steps > 1:
                self.data['log_return'].append(math.log(self.data['mid_price'][-1] / self.data['mid_price'][-2]))

    def strategy(self):
        if self.current_steps < self.vol_window + 1:
            return

        self.k = self.ks[self.current_steps]
        vol = np.std(self.data['log_return'][-self.vol_window:]) ** 2
        s = self.data['mid_price'][-1]
        q = self.state.position.get(self.product, 0)
        r = s - q * self.y * vol * (self.max_t - self.current_steps)
        spread = self.y * vol * (self.max_t - self.current_steps) + (2 / self.y) * math.log(1 + self.y / self.k)
        bid = r - spread / 2
        ask = r + spread / 2

        print(f'Bid: {bid}, Ask: {ask}')

        if s < 20:
            self.orders[self.product].append(Order(self.product, bid, self.limit - s))

        if s > -20:
            self.orders[self.product].append(Order(self.product, ask, -(self.limit - s)))


class BolBStrategy(BaseStrategy):
    def __init__(self, smoothing: int = 20, stds: int = 2):
        super().__init__()
        self.smoothing = smoothing
        self.stds = stds

        self.product = None
        self.current_steps = 0
        self.data = {}

    def accumulate(self):
        if self.product in self.state.order_depths.keys():
            depth = self.state.order_depths[self.product]
            self.data['top_bid'] = max(depth.buy_orders.keys())
            self.data['top_ask'] = min(depth.sell_orders.keys())
            self.data['mid_price'] = self.data['mid_price'].get(
                self.current_steps, []) + [(self.data['top_bid'] + self.data['top_ask']) / 2]
            self.current_steps += 1

    def strategy(self):
        if self.current_steps < self.smoothing:
            return

        prices = self.data['mid_price'][self.current_steps - self.smoothing:]
        bolu = np.mean(prices) + self.stds * np.std(prices)
        bold = np.mean(prices) - self.stds * np.std(prices)

        if prices[-1] > bolu:
            self.orders[self.product] = [
                Order(self.product, self.data['top_bid'], -1)]

        elif prices[-1] < bold:
            self.orders[self.product] = [
                Order(self.product, self.data['top_ask'], 1)]


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
        print(vars(self.state))

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


class BaseTrader:
    def __init__(self, strategies: Dict[str, BaseStrategy]):
        self.strategies = strategies

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}

        for strategy in self.strategies.values():
            strategy_out = strategy(state)

            for product, orders in strategy_out.items():
                if product not in result.keys():
                    result[product] = result.get(product, []) + orders

        return result


class Trader(BaseTrader):
    def __init__(self):
        super().__init__(
            {'PEARLS': AvellanedaMM(
                'PEARLS', 0.01, 100, 20000,
            )
            }
        )
