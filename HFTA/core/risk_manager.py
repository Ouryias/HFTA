# HFTA/core/risk_manager.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from HFTA.broker.client import Quote, PortfolioSnapshot
from HFTA.strategies.base import OrderIntent

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """
    Very simple risk configuration.

    - max_notional_per_order: absolute cap per order (e.g. 100.0 = $100)
    - max_cash_utilization: fraction of current cash that a single BUY can use
      (e.g. 0.1 = 10% of available cash)
    """
    max_notional_per_order: float = 100.0
    max_cash_utilization: float = 0.1


class RiskManager:
    """
    Stateless per-order risk checks.
    Later we can extend this with:
      - daily PnL tracking
      - per-strategy limits
      - global kill-switch, etc.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def _infer_price(self, oi: OrderIntent, quote: Quote) -> Optional[float]:
        # Prefer explicit limit, otherwise use last/ask/bid in that order
        if oi.limit_price is not None:
            return oi.limit_price
        if quote.last is not None:
            return quote.last
        if oi.side.lower() == "buy" and quote.ask is not None:
            return quote.ask
        if oi.side.lower() == "sell" and quote.bid is not None:
            return quote.bid
        return None

    def approve(self, oi: OrderIntent, quote: Quote, snapshot: PortfolioSnapshot) -> bool:
        price = self._infer_price(oi, quote)
        if price is None:
            logger.info("Risk: rejecting %s (no usable price)", oi)
            return False

        notional = price * oi.quantity

        # Hard cap per order
        if notional > self.config.max_notional_per_order:
            logger.info(
                "Risk: rejecting %s (notional %.2f > max_notional_per_order %.2f)",
                oi, notional, self.config.max_notional_per_order,
            )
            return False

        # Simple cash check for BUYs
        if oi.side.lower() == "buy":
            max_allowed = snapshot.cash_available * self.config.max_cash_utilization
            if notional > max_allowed:
                logger.info(
                    "Risk: rejecting %s (notional %.2f > cash_allowed %.2f)",
                    oi, notional, max_allowed,
                )
                return False

        # SELLs: for now we assume you are not shorting (Wealthsimple TFSA/non-margin).
        # More advanced position checks can be added later.

        return True
