# HFTA/core/order_manager.py

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Mapping

from HFTA.broker.client import WealthsimpleClient, Quote, PortfolioSnapshot
from HFTA.core.execution_tracker import ExecutionTracker
from HFTA.core.risk_manager import RiskManager
from HFTA.strategies.base import OrderIntent

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Central place that:
    - Receives OrderIntent objects from strategies
    - Asks RiskManager if they are allowed
    - Records fills in ExecutionTracker
    - If live=True, sends them via WealthsimpleClient
    """

    def __init__(
        self,
        client: WealthsimpleClient,
        risk_manager: RiskManager,
        execution_tracker: Optional[ExecutionTracker] = None,
        live: bool = False,
    ) -> None:
        self.client = client
        self.risk_manager = risk_manager
        self.execution_tracker = execution_tracker
        self.live = live

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _infer_price(self, oi: OrderIntent, quote: Quote) -> Optional[float]:
        """
        Pick a reasonable execution price for PnL tracking.
        """
        if oi.limit_price is not None:
            return float(oi.limit_price)

        if quote.last is not None:
            return float(quote.last)

        side = oi.side.lower()
        if side == "buy" and quote.ask is not None:
            return float(quote.ask)
        if side == "sell" and quote.bid is not None:
            return float(quote.bid)

        return None

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    def process_order(
        self,
        oi: OrderIntent,
        quote: Quote,
        snapshot: PortfolioSnapshot,
        positions: Mapping[str, Any],
    ) -> None:
        if not self.risk_manager.approve(oi, quote, snapshot, positions):
            logger.info("Order blocked by risk: %s", oi)
            return

        logger.info("Order approved: %s (live=%s)", oi, self.live)

        # Determine a price for tracking (paper fills in DRY-RUN, approx in live)
        price = self._infer_price(oi, quote)
        if price is None:
            logger.info("Skipping PnL tracking for %s (no usable price)", oi)
        elif self.execution_tracker is not None:
            self.execution_tracker.record_fill(oi, price, quote.timestamp)

        # Live mode: actually send to broker
        if not self.live:
            return

        self.client.place_equity_order(
            symbol=oi.symbol,
            side=oi.side,
            quantity=oi.quantity,
            order_type=oi.order_type,
            limit_price=oi.limit_price,
        )
