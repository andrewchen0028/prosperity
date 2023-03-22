from trader import *
import pandas as pd


class Backtester:
    def __init__(self, trader, orderbook, product):
        self.trader = trader
        self.orderbook = orderbook
        self.position = {
            product: 0
        }
        self.state = None

    def get_trade_state(self, step):
        order_depth = OrderDepth()
        order_depth.buy_orders = dict(self.orderbook[step, [2, 4, 6]], self.orderbook[step, [3, 5, 7]])
        order_depth.sell_orders = dict(self.orderbook[step, [8, 10, 12]], self.orderbook[step, [9, 11, 13]])
        self.state = TradingState(
            timestamp=step,
            listings=None,
            order_depths={self.product: order_depth},
            own_trades=None,
            market_trades=None,
            position=self.position,
            observations=None,
        )

    def run(self, trader):
        for step in range(len(self.orderbook)):
            self.get_trade_state(step)
            orders = trader.run(self.state)


        return self.orderbook