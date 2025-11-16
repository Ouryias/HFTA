# HFTA/strategies/base.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from HFTA.broker.client import Quote


@dataclass
class OrderIntent:
    symbol: str
    side: str          # "buy" or "sell"
    quantity: float
    order_type: str    # "limit", "market", etc.
    limit_price: Optional[float] = None
    meta: Dict[str, Any] = None


class Strategy:
    """
    Base class for all strategies.
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.config = config

    def on_quote(self, quote: Quote) -> List[OrderIntent]:
        """
        Called on each quote update. Return a list of order intents.
        """
        raise NotImplementedError
