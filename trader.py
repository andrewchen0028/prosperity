"""
Supported Libraries:
 - pandas
 - numpy
 - statistics
 - math
 - typing
"""

from math import ceil, floor
from typing import Dict, List
from datamodel import TradingState, Order


# DONE: Make buy/sell wall at top of book
#       Bid just below fair value if we are short.
#       Ask just above fair value if we are long.
# (unreasoned heuristic to keep balanced book)
# NOTE: This change improved PnL from 1546 to 1605

# TODO: Try changing the market-making bid/ask prices


class Trader:
    pearls_ema = None
    pearls_limit = 20

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        result = {}

        if self.pearls_ema:
            print(str(round(self.pearls_ema, 2)).rjust(8, " "), state.position)

        for product in state.order_depths.keys():
            if product == "PEARLS":
                # Mean reversion strategy
                bids: Dict[int, int] = state.order_depths[product].buy_orders
                asks: Dict[int, int] = state.order_depths[product].sell_orders
                print("bids: ", bids)
                print("asks: ", asks)
                position: int = state.position["PEARLS"] \
                    if "PEARLS" in state.position.keys() else 0
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
                mid_price = (best_bid + best_ask) / 2

                # Update EMA20
                N = 20
                k = 2 / (N + 1)
                self.pearls_ema = mid_price * k + self.pearls_ema * (1 - k) \
                    if self.pearls_ema else mid_price

                # TODO: The following two if statements don't yet handle the situation
                #           (best ask volume) < (remainder of position limit).
                #       So they won't fill the "second-best bid/ask", even below EMA20.
                #       Make it do that.
                # NOTE: positive quantity => buy order

                # Fill asks below EMA20
                if position < self.pearls_limit and best_ask < self.pearls_ema:
                    qty = max(position - self.pearls_limit,
                              best_ask_volume)  # NEGATIVE
                    print("best_ask_volume: ", best_ask_volume)
                    print("BUY", str(-qty), "at", best_ask)
                    orders.append(Order(product, best_ask, -qty))

                    if position + abs(qty) > 0:
                        # We will be long pearls after this purchase.
                        # Ask just above fair value.
                        orders.append(
                            Order(product, ceil(self.pearls_ema), -(position + abs(qty))))
                        print("ASK", str(-(position + abs(qty))),
                              "at", ceil(self.pearls_ema))
                    elif position + abs(qty) < 0:
                        # We will be short pearls after this purchase.
                        # Bid just below fair value.
                        orders.append(
                            Order(product, floor(self.pearls_ema), -(position + abs(qty))))
                        print("BID", str(-(position + abs(qty))),
                              "at", floor(self.pearls_ema))

                # Fill bids above EMA20
                if position > -self.pearls_limit and best_bid > self.pearls_ema:
                    qty = min(position + self.pearls_limit,
                              best_bid_volume)  # POSITIVE
                    print("best_bid_volume: ", best_bid_volume)
                    print("SELL", str(qty), "at", best_bid)
                    orders.append(Order(product, best_bid, -qty))

                    if position - abs(qty) > 0:
                        # We will be long pearls after this sale.
                        # Ask just above fair value.
                        orders.append(
                            Order(product, ceil(self.pearls_ema), -(position - abs(qty))))
                        print("ASK", str(-(position - abs(qty))),
                              "at", ceil(self.pearls_ema))
                    elif position - abs(qty) < 0:
                        # We will be short pearls after this sale.
                        # Bid just below fair value.
                        orders.append(
                            Order(product, floor(self.pearls_ema), -(position - abs(qty))))
                        print("BID", str(-(position - abs(qty))),
                              "at", floor(self.pearls_ema))

                # Add orders to result dict
                result[product] = orders

        print()
        return result
