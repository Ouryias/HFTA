# HFTA/core/engine.py

from __future__ import annotations

import logging
import time
from typing import List

from HFTA.broker.client import WealthsimpleClient
from HFTA.core.order_manager import OrderManager
from HFTA.strategies.base import Strategy, OrderIntent

logger = logging.getLogger(__name__)


class Engine:
    """
    Very simple polling engine:
    - Polls quotes for a list of symbols
    - Feeds them into all strategies
    - Passes the resulting orders to OrderManager
    """

    def __init__(
        self,
        client: WealthsimpleClient,
        strategies: List[Strategy],
        symbols: List[str],
        order_manager: OrderManager,
        poll_interval: float = 2.0,
    ) -> None:
        self.client = client
        self.strategies = strategies
        self.symbols = [s.upper() for s in symbols]
        self.order_manager = order_manager
        self.poll_interval = poll_interval

    def run_forever(self) -> None:
        """
        Main loop. Gracefully stops on KeyboardInterrupt (Ctrl+C).
        """
        logger.info("Engine loop starting (live=%s)", self.order_manager.live)
        try:
            while True:
                # One portfolio snapshot per loop
                snapshot = self.client.get_portfolio_snapshot()

                for sym in self.symbols:
                    quote = self.client.get_quote(sym)
                    logger.debug("Quote: %s", quote)

                    for strat in self.strategies:
                        intents = strat.on_quote(quote)
                        for oi in intents:
                            self.order_manager.process_order(oi, quote, snapshot)

                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            logger.info("Engine stopped by user (KeyboardInterrupt).")
