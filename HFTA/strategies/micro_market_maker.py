# HFTA/strategies/micro_market_maker.py

from __future__ import annotations
from typing import Any, Dict, List
from HFTA.strategies.base import Strategy, OrderIntent
from HFTA.broker.client import Quote


class MicroMarketMaker(Strategy):
    """
    Simple single-symbol market maker:
    - Posts bid and ask around current mid
    - Keeps inventory within +/- max_inventory
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        super().__init__(name, config)
        self.symbol = config["symbol"]
        self.max_inventory = config.get("max_inventory", 5)
        self.spread = config.get("spread", 0.05)          # absolute price width around mid
        self.order_quantity = config.get("order_quantity", 1)
        self.position = 0.0

    def update_position(self, new_position: float) -> None:
        self.position = new_position

    def on_quote(self, quote: Quote) -> List[OrderIntent]:
        if quote.symbol != self.symbol or quote.bid is None or quote.ask is None:
            return []

        mid = (quote.bid + quote.ask) / 2.0
        bid_price = round(mid - self.spread, 2)
        ask_price = round(mid + self.spread, 2)

        intents: List[OrderIntent] = []

        # Buy side if inventory below max
        if self.position < self.max_inventory:
            intents.append(
                OrderIntent(
                    symbol=self.symbol,
                    side="buy",
                    quantity=self.order_quantity,
                    order_type="limit",
                    limit_price=bid_price,
                    meta={"strategy": self.name},
                )
            )

        # Sell side if inventory above -max
        if self.position > -self.max_inventory:
            intents.append(
                OrderIntent(
                    symbol=self.symbol,
                    side="sell",
                    quantity=self.order_quantity,
                    order_type="limit",
                    limit_price=ask_price,
                    meta={"strategy": self.name},
                )
            )

        return intents
