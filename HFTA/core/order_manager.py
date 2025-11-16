# HFTA/core/order_manager.py

from __future__ import annotations

import logging

from HFTA.broker.client import WealthsimpleClient, Quote, PortfolioSnapshot
from HFTA.core.risk_manager import RiskManager
from HFTA.strategies.base import OrderIntent

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Central place that:
    - Receives OrderIntent objects from strategies
    - Asks RiskManager if they are allowed
    - If approved and live=True, sends them via WealthsimpleClient
    """

    def __init__(
        self,
        client: WealthsimpleClient,
        risk_manager: RiskManager,
        live: bool = False,
    ) -> None:
        self.client = client
        self.risk_manager = risk_manager
        self.live = live

    def process_order(
        self,
        oi: OrderIntent,
        quote: Quote,
        snapshot: PortfolioSnapshot,
    ) -> None:
        if not self.risk_manager.approve(oi, quote, snapshot):
            logger.info("Order blocked by risk: %s", oi)
            return

        logger.info("Order approved: %s (live=%s)", oi, self.live)

        if not self.live:
            # Dry-run: do nothing else
            return

        # Live mode: actually send to broker
        self.client.place_equity_order(
            symbol=oi.symbol,
            side=oi.side,
            quantity=oi.quantity,
            order_type=oi.order_type,
            limit_price=oi.limit_price,
        )
