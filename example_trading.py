from datamodel import Listing, OrderDepth, Trade, TradingState

from trader_example import Trader

timestamp = 1000

listings = {
    "PEARLS": Listing(symbol="PEARLS", product="PEARLS", denomination="SEASHELLS"),
    "BANANAS": Listing(symbol="BANANAS", product="BANANAS", denomination="SEASHELLS"),
}

order_depths = {
    "PEARLS": OrderDepth(buy_orders={10: 7, 9: 5},
                         sell_orders={11: -4, 12: -8}),
    "BANANAS": OrderDepth(buy_orders={142: 3, 141: 5},
                          sell_orders={144: -5, 145: -8}),
}

own_trades = {"PEARLS": [], "BANANAS": []}

market_trades = {
    "PEARLS": [Trade(symbol="PEARLS", price=11, quantity=4, buyer="", seller="")],
    "BANANAS": []
}

position = {"PEARLS": 3, "BANANAS": -5}

observations = {}

state = TradingState(timestamp=timestamp,
                     listings=listings,
                     orderbook=order_depths,
                     own_trades=own_trades,
                     market_trades=market_trades,
                     position={},
                     observations={})

trader = Trader()
print(trader.run(state=state))
