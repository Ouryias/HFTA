# HFTA/core/engine.py

from __future__ import annotations

import logging
import time
from typing import List, Optional, Mapping, Any, Dict

from HFTA.broker.client import WealthsimpleClient, PortfolioSnapshot, Quote
from HFTA.core.order_manager import OrderManager
from HFTA.strategies.base import Strategy
from HFTA.market.quote_provider import BaseQuoteProvider
from HFTA.market.intraday_stats import IntradayStatsTracker
from HFTA.symbol_selection import SymbolSelector

logger = logging.getLogger(__name__)


class Engine:
    """Main event loop for strategy execution.

    Responsibilities
    ----------------
    - Pull portfolio snapshot and positions from broker (or a cached view).
    - Fetch all required quotes via the configured quote provider.
    - Dispatch quotes to strategies and collect OrderIntents.
    - Route resulting OrderIntents through the OrderManager.
    - Optionally run the AI controller and dynamic symbol picker each loop.

    DRY-RUN mode
    ------------
    When `order_manager.live` is False and `paper_cash` is provided, we:
      - Fetch holdings from Wealthsimple once at startup (to seed positions).
      - Override the portfolio snapshot cash/net_worth with `paper_cash`.
      - Use ExecutionTracker positions for risk checks, so that risk limits
        reflect simulated fills within the session without hammering the DB.
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
        # Symbols configured explicitly (e.g. from JSON config)
        self._configured_symbols: List[str] = [s.upper() for s in symbols]
        self.symbols: List[str] = list(self._configured_symbols)

        self.order_manager = order_manager
        self.quote_provider = quote_provider
        self.poll_interval = float(poll_interval)
        self.paper_cash = paper_cash
        self.ai_controller = ai_controller
        self.intraday_stats = intraday_stats
        self.symbol_selector = symbol_selector

        # Cached broker state for DRY-RUN
        self._initial_snapshot: Optional[PortfolioSnapshot] = None
        self._initial_positions: Optional[Mapping[str, Any]] = None

        # Strategy routing by symbol
        self._strategies_by_symbol: Dict[str, List[Strategy]] = {}
        self._symbol_agnostic_strategies: List[Strategy] = []
        self._last_symbol_set: Optional[set[str]] = None
        self._rebuild_symbol_index()

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
    # Strategy / symbol routing
    # ------------------------------------------------------------------ #

    def _rebuild_symbol_index(self) -> None:
        """
        Rebuild internal mapping:
            symbol -> [strategies bound to that symbol]
        and the list of symbol-agnostic strategies.

        Also updates `self.symbols` to be the union of:
          - symbols configured explicitly, and
          - any `strategy.symbol` attributes that are set.
        """
        symbol_set: set[str] = {s.upper() for s in self._configured_symbols}

        self._strategies_by_symbol.clear()
        self._symbol_agnostic_strategies = []

        for strat in self.strategies:
            sym = getattr(strat, "symbol", None)
            if sym is None:
                self._symbol_agnostic_strategies.append(strat)
                continue
            sym_u = str(sym).upper()
            symbol_set.add(sym_u)
            self._strategies_by_symbol.setdefault(sym_u, []).append(strat)

        new_symbol_set = symbol_set
        if self._last_symbol_set is None or new_symbol_set != self._last_symbol_set:
            logger.info("Engine: active symbol universe updated to %s", sorted(new_symbol_set))
            self._last_symbol_set = set(new_symbol_set)

        self.symbols = sorted(new_symbol_set)

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

                # 0) Ensure we have up-to-date strategy/symbol routing
                self._rebuild_symbol_index()

                # 1) Portfolio snapshot + holdings from broker
                tracker = getattr(self.order_manager, "execution_tracker", None)

                use_live_state = self.order_manager.live or self.paper_cash is None

                if use_live_state or self._initial_snapshot is None or self._initial_positions is None:
                    # Live mode, or first loop in DRY-RUN: hit the broker.
                    real_snapshot = self.client.get_portfolio_snapshot()
                    ws_positions = self.client.get_equity_positions()

                    if tracker is not None:
                        # Seed tracker once with live holdings; subsequent calls
                        # are cheap no-ops thanks to internal guard.
                        tracker.seed_from_positions(ws_positions)

                    # Cache for DRY-RUN.
                    if not self.order_manager.live and self.paper_cash is not None:
                        if self._initial_snapshot is None:
                            self._initial_snapshot = real_snapshot
                        if self._initial_positions is None:
                            self._initial_positions = ws_positions
                else:
                    # DRY-RUN: reuse cached snapshot/positions to avoid repeated
                    # DB/HTTP work (Peewee queries, API calls).
                    real_snapshot = self._initial_snapshot
                    ws_positions = self._initial_positions

                if real_snapshot is None or ws_positions is None:
                    logger.error("Engine: missing snapshot or positions; aborting loop.")
                    time.sleep(self.poll_interval)
                    continue

                snapshot = self._make_sim_snapshot(real_snapshot)
                positions_for_risk: Mapping[str, Any] = self._positions_for_risk(ws_positions)

                # 2) Fetch all quotes for the current symbol list via provider
                if not self.symbols:
                    logger.warning("Engine loop %d: no active symbols configured.", loop_idx)
                    time.sleep(self.poll_interval)
                    continue

                quotes_by_symbol: Mapping[str, Quote] = self.quote_provider.get_quotes(self.symbols)
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

                    # Symbol-specific strategies plus symbol-agnostic ones
                    strategies_for_symbol = self._strategies_by_symbol.get(sym, [])
                    if self._symbol_agnostic_strategies:
                        strategies_for_symbol = strategies_for_symbol + self._symbol_agnostic_strategies

                    for strat in strategies_for_symbol:
                        intents = strat.on_quote(quote)
                        for oi in intents:
                            self.order_manager.process_order(
                                oi, quote, snapshot, positions_for_risk
                            )
                            # In DRY-RUN, refresh risk view to include any new
                            # simulated fills before the next order.
                            if tracker is not None and not self.order_manager.live:
                                positions_for_risk = self._positions_for_risk(ws_positions)

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
                        # Any changes to strategy.symbol will be picked up
                        # on the next loop via _rebuild_symbol_index().
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
