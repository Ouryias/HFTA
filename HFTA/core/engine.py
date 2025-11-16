# HFTA/core/engine.py

from __future__ import annotations

import logging
import time
from typing import List

from HFTA.broker.client import WealthsimpleClient
from HFTA.strategies.base import Strategy, OrderIntent

logger = logging.getLogger(__name__)


class Engine:
    """
    Very simple polling engine:
    - Polls quotes for a list of symbols
    - Feeds them into all strategies
    - Sends the resulting orders via WealthsimpleClient
    """

    def __init__(
        self,
        client: WealthsimpleClient,
        strategies: List[Strategy],
        symbols: List[str],
        poll_interval: float = 2.0,
        live: bool = False,
    ) -> None:
        self.client = client
        self.strategies = strategies
        self.symbols = [s.upper() for s in symbols]
        self.poll_interval = poll_interval
        self.live = live

    def _handle_order(self, oi: OrderIntent) -> None:
        logger.info("OrderIntent: %s", oi)
        if not self.live:
            # Dry-run mode: do not actually send orders
            return

        self.client.place_equity_order(
            symbol=oi.symbol,
            side=oi.side,
            quantity=oi.quantity,
            order_type=oi.order_type,
            limit_price=oi.limit_price,
        )

    def run_forever(self) -> None:
        """
        Main loop. Gracefully stops on KeyboardInterrupt (Ctrl+C).
        """
        logger.info("Engine loop starting (live=%s)", self.live)
        try:
            while True:
                for sym in self.symbols:
                    quote = self.client.get_quote(sym)
                    logger.debug("Quote: %s", quote)

                    for strat in self.strategies:
                        intents = strat.on_quote(quote)
                        for oi in intents:
                            self._handle_order(oi)

                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Engine stopped by user (KeyboardInterrupt).")
