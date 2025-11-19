# HFTA/core/engine.py

from __future__ import annotations

import logging
import time
from typing import List, Optional, Mapping, Any

from HFTA.broker.client import WealthsimpleClient, PortfolioSnapshot, Quote
from HFTA.core.order_manager import OrderManager
from HFTA.strategies.base import Strategy
from HFTA.market.quote_provider import BaseQuoteProvider
from HFTA.market.intraday_stats import IntradayStatsTracker
from HFTA.symbol_selection import SymbolSelector

logger = logging.getLogger(__name__)


class Engine:
    """Main event loop for strategy execution.

    Responsibilities:
      - Pull portfolio snapshot and positions from broker.
      - Fetch all required quotes via the configured quote provider.
      - Dispatch quotes to all strategies and collect OrderIntents.
      - Route resulting OrderIntents through the OrderManager.
      - Optionally run the AI controller and dynamic symbol picker each loop.

    In DRY-RUN mode (live=False with paper_cash set), the portfolio snapshot
    is overridden with the configured paper_cash while still using the real
    holdings structure for sizing/risk.
    """

    def __init__(
        self,
        client: WealthsimpleClient,
        strategies: List[Strategy],
        symbols: List[str],
        order_manager: OrderManager,
        quote_provider: BaseQuoteProvider,
        poll_interval: float = 2.0,
        paper_cash: Optional[float] = None,
        ai_controller: Optional[Any] = None,
        intraday_stats: Optional[IntradayStatsTracker] = None,
        symbol_selector: Optional[SymbolSelector] = None,
    ) -> None:
        self.client = client
        self.strategies = strategies
        self.symbols = [s.upper() for s in symbols]
        self.order_manager = order_manager
        self.quote_provider = quote_provider
        self.poll_interval = poll_interval
        self.paper_cash = paper_cash
        self.ai_controller = ai_controller
        self.intraday_stats = intraday_stats
        self.symbol_selector = symbol_selector

    # ------------------------------------------------------------------ #
    # DRY-RUN snapshot helpers
    # ------------------------------------------------------------------ #

    def _make_sim_snapshot(self, snapshot: PortfolioSnapshot) -> PortfolioSnapshot:
        """Override net worth / cash with paper_cash in DRY-RUN mode.

        - In live mode (order_manager.live == True) or when paper_cash is None,
          the broker snapshot is passed through unchanged.
        """
        if self.order_manager.live or self.paper_cash is None:
            return snapshot

        return PortfolioSnapshot(
            account_id=snapshot.account_id,
            currency=snapshot.currency,
            net_worth=self.paper_cash,
            cash_available=self.paper_cash,
        )

    def _positions_for_risk(self, ws_positions: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the positions view that the RiskManager should see.

        - Live mode: use Wealthsimple holdings.
        - DRY-RUN   : use ExecutionTracker summary to reflect simulated fills.
        """
        tracker = getattr(self.order_manager, "execution_tracker", None)
        if self.order_manager.live or tracker is None:
            return ws_positions
        return tracker.summary()

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def run_forever(self) -> None:
        """Run the engine loop until interrupted.

        This method blocks in an infinite loop and should typically be run
        from a script entrypoint (see scripts/run_engine.py).
        """
        loop_idx = 0
        logger.info(
            "Engine loop starting (live=%s, paper_cash=%s)",
            self.order_manager.live,
            self.paper_cash,
        )

        try:
            while True:
                loop_idx += 1
                loop_start = time.time()

                # 1) Portfolio snapshot + holdings from broker
                real_snapshot = self.client.get_portfolio_snapshot()
                ws_positions = self.client.get_equity_positions()

                tracker = getattr(self.order_manager, "execution_tracker", None)
                if tracker is not None:
                    tracker.seed_from_positions(ws_positions)

                snapshot = self._make_sim_snapshot(real_snapshot)
                positions_for_risk = self._positions_for_risk(ws_positions)

                # 2) Fetch all quotes for the current symbol list via provider
                quotes_by_symbol: Mapping[str, Quote] = self.quote_provider.get_quotes(
                    self.symbols
                )
                if not quotes_by_symbol:
                    logger.warning(
                        "Engine loop %d: no quotes returned for symbols=%s",
                        loop_idx,
                        self.symbols,
                    )

                # 3) Run strategies on each quote (and update intraday stats)
                for sym in self.symbols:
                    quote = quotes_by_symbol.get(sym)
                    if quote is None:
                        logger.debug(
                            "Engine loop %d: missing quote for %s; skipping.",
                            loop_idx,
                            sym,
                        )
                        continue

                    logger.debug("Quote: %s", quote)

                    # Feed intraday stats tracker, if enabled
                    if self.intraday_stats is not None:
                        price: Optional[float] = quote.last
                        if price is None:
                            if quote.bid is not None and quote.ask is not None:
                                price = (quote.bid + quote.ask) / 2.0
                            elif quote.bid is not None:
                                price = quote.bid
                            elif quote.ask is not None:
                                price = quote.ask
                        if price is not None:
                            self.intraday_stats.on_quote(sym, price)

                    for strat in self.strategies:
                        intents = strat.on_quote(quote)
                        for oi in intents:
                            self.order_manager.process_order(
                                oi, quote, snapshot, positions_for_risk
                            )

                # 4) AI controller can adjust strategies/risk each loop
                if self.ai_controller is not None and tracker is not None:
                    try:
                        self.ai_controller.on_loop(
                            strategies=self.strategies,
                            risk_config=self.order_manager.risk_manager.config,
                            tracker=tracker,
                        )
                    except Exception:
                        logger.exception("AIController.on_loop failed")

                # 5) Dynamic symbol selector (market-wide picker)
                if self.symbol_selector is not None:
                    try:
                        self.symbol_selector.on_loop(
                            strategies=self.strategies,
                            tracker=tracker,
                            intraday_stats=self.intraday_stats,
                        )
                    except Exception:
                        logger.exception("SymbolSelector.on_loop failed")

                if tracker is not None:
                    tracker.log_summary()

                loop_end = time.time()
                elapsed = loop_end - loop_start
                logger.debug("Engine loop %d took %.4fs", loop_idx, elapsed)

                # Sleep the remainder of poll_interval (never negative)
                sleep_for = max(self.poll_interval - elapsed, 0.0)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except KeyboardInterrupt:
            logger.info("Engine stopped by user (KeyboardInterrupt).")
