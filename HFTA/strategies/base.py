# HFTA/strategies/base.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from HFTA.broker.client import Quote


@dataclass
class OrderIntent:
    """
    Simple container describing an order request emitted by a Strategy.

    Fields
    ------
    symbol:
        Ticker symbol (e.g. "AAPL").
    side:
        "buy" or "sell".
    quantity:
        Number of shares to trade (positive).
    order_type:
        Order type understood by the broker client (e.g. "limit", "market").
    strategy_name:
        Optional logical name of the strategy that created this intent.
        If not set, OrderManager / ExecutionTracker will often fall back
        to the Strategy.name.
    limit_price:
        Limit price for limit orders. Can be None for market orders.
    meta:
        Optional free-form dict for extra information (reason, debug tags, etc.).
    """
    symbol: str
    side: str          # "buy" or "sell"
    quantity: float
    order_type: str    # "limit", "market", etc.
    # Optional fields
    strategy_name: Optional[str] = None
    limit_price: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Normalize basic fields a bit for consistency in logs.
        self.symbol = self.symbol.upper()
        self.side = self.side.lower()
        if self.quantity < 0:
            # Keep quantity non-negative, encode sign via side.
            self.quantity = abs(self.quantity)

    def __repr__(self) -> str:
        parts = [
            f"symbol={self.symbol}",
            f"side={self.side}",
            f"qty={self.quantity:.4f}",
            f"type={self.order_type}",
        ]
        if self.limit_price is not None:
            parts.append(f"limit={self.limit_price:.4f}")
        if self.strategy_name:
            parts.append(f"strategy={self.strategy_name}")
        return f"OrderIntent({', '.join(parts)})"


class Strategy:
    """
    Base class for all strategies.

    A Strategy receives quotes and emits zero or more OrderIntent objects.

    Subclasses must implement:
        - on_quote(self, quote: Quote) -> List[OrderIntent]
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        # Make a shallow copy so strategies can mutate config without
        # affecting the original dict passed from config loaders.
        self.config: Dict[str, Any] = dict(config or {})

        # Expose config keys as attributes for convenience, but do not
        # override attributes that subclasses may already have set.
        for key, value in self.config.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def on_quote(self, quote: Quote) -> List[OrderIntent]:
        """
        Called on each quote update. Should return a list of OrderIntent.
        """
        raise NotImplementedError
