from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


class Trader:
    """
    Example implementation, probably not profitable
    """

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the keys (the available products) contained in the orderbook
        for product in state.order_depths.keys():
            if product == 'BTC':
                order_depths: OrderDepth = state.order_depths[product]
                orders: list[Order] = []
                fair_value = 13

                if order_depths.sell_orders:
                    best_ask = min(order_depths.sell_orders.keys())
                    best_ask_volume = order_depths.sell_orders[best_ask]

                    if best_ask < fair_value:
                        print("BUY", str(-best_ask_volume) + "x", best_ask)
                        orders.append(
                            Order(product, best_ask, -best_ask_volume))

                # The below code block is similar to the one above,
                # the difference is that it finds the highest bid (buy order)
                # If the price of the order is higher than the fair value
                # This is an opportunity to sell at a premium
                if order_depths.buy_orders:
                    best_bid = max(order_depths.buy_orders.keys())
                    best_bid_volume = order_depths.buy_orders[best_bid]
                    if best_bid > fair_value:
                        print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(
                            Order(product, best_bid, -best_bid_volume))

                # Add all the above orders to the result dict
                result[product] = orders

        # Return the dict of orders
        # These possibly contain buy or sell orders for PEARLS
        # Depending on the logic above
        return result
