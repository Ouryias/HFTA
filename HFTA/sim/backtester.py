# HFTA/sim/backtester.py

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict

from HFTA.broker.client import Quote, PortfolioSnapshot
from HFTA.core.execution_tracker import ExecutionTracker, PositionState
from HFTA.core.order_manager import OrderManager
from HFTA.core.risk_manager import RiskConfig, RiskManager
from HFTA.strategies.base import Strategy


@dataclass
class BacktestConfig:
    """
    Configuration for an offline backtest.
    Controls the synthetic market and starting account state.
    """
    symbol: str = "AAPL"
    starting_price: float = 40.0
    starting_cash: float = 100_000.0
    steps: int = 2_000
    step_seconds: int = 5
    volatility_annual: float = 0.4
    spread_cents: float = 0.10
    risk_config: Optional[RiskConfig] = None


@dataclass
class BacktestResult:
    symbol: str
    starting_cash: float
    final_cash: float
    final_equity: float
    realized_pnl: float
    max_drawdown: float
    equity_curve: List[float]
    timestamps: List[datetime]
    positions_summary: Dict[str, PositionState]


def generate_random_walk_quotes(
    symbol: str,
    starting_price: float,
    steps: int,
    step_seconds: int,
    volatility_annual: float,
    spread_cents: float,
    start_time: Optional[datetime] = None,
) -> List[Quote]:
    """
    Generate List[Quote] following geometric Brownian motion:

        dS/S = sigma * sqrt(dt) * N(0,1)

    - Start at `starting_price`
    - `steps` quotes, spaced `step_seconds` apart
    - Fixed bid/ask spread of `spread_cents` around the mid
    """
    if steps <= 0:
        return []

    if start_time is None:
        start_time = datetime.utcnow()

    dt_years = step_seconds / (365.0 * 24.0 * 3600.0)
    sigma = float(volatility_annual)

    price = float(starting_price)
    quotes: List[Quote] = []

    for i in range(steps):
        # GBM step
        z = random.gauss(0.0, 1.0)
        price *= math.exp(sigma * math.sqrt(dt_years) * z)

        mid = max(price, 0.01)  # guard against weird negatives
        half_spread = spread_cents / 2.0
        bid = mid - half_spread
        ask = mid + half_spread

        if bid < 0.01:
            bid = 0.01
        if ask <= bid:
            ask = bid + spread_cents

        ts = start_time + timedelta(seconds=i * step_seconds)
        ts_iso = ts.isoformat()

        q = Quote(
            symbol=symbol.upper(),
            security_id=f"SIM-{symbol.upper()}",
            bid=round(bid, 4),
            ask=round(ask, 4),
            last=round(mid, 4),
            bid_size=None,
            ask_size=None,
            timestamp=ts_iso,
        )
        quotes.append(q)

    return quotes


class BacktestEngine:
    """
    Offline backtest engine.

    - Generates or accepts a list of Quotes.
    - Reuses RiskManager, OrderManager, ExecutionTracker like Engine.run_forever,
      but with:
        * client=None
        * live=False (no Wealthsimple calls)
    - Tracks cash, equity curve, max drawdown.
    """

    def __init__(
        self,
        strategies: List[Strategy],
        config: BacktestConfig,
        quotes: Optional[List[Quote]] = None,
    ) -> None:
        self.strategies = strategies
        self.config = config
        self.quotes = quotes

        self.tracker = ExecutionTracker()
        rc = config.risk_config or RiskConfig()
        self.risk_manager = RiskManager(rc)

        # client is never touched if live=False, so None is fine.
        self.order_manager = OrderManager(
            client=None,  # type: ignore[arg-type]
            risk_manager=self.risk_manager,
            execution_tracker=self.tracker,
            live=False,
        )

        # ExecutionTracker tracks positions + realized PnL; we track cash.
        self.starting_cash = float(config.starting_cash)

    # ---------------- helpers ---------------- #

    def _recompute_cash(self) -> float:
        """
        Cash = starting_cash - sum(buys) + sum(sells) across all fills.
        """
        cash = self.starting_cash
        for f in self.tracker.fills:
            notional = f.price * f.quantity
            if f.side == "buy":
                cash -= notional
            elif f.side == "sell":
                cash += notional
        return cash

    def _equity(self, mid_price: float, cash: float) -> float:
        """
        Equity = cash + mark-to-market value of all open positions at mid_price.
        (Single-symbol backtest, but we sum anyway.)
        """
        value = 0.0
        for pos in self.tracker.positions.values():
            value += pos.quantity * mid_price
        return cash + value

    def _make_snapshot(self, equity: float, cash: float) -> PortfolioSnapshot:
        """
        Fake PortfolioSnapshot for RiskManager.approve.
        """
        return PortfolioSnapshot(
            account_id="BACKTEST",
            currency="SIM",
            net_worth=equity,
            cash_available=cash,
        )

    # ---------------- main entry ---------------- #

    def run(self) -> BacktestResult:
        cfg = self.config

        quotes = self.quotes or generate_random_walk_quotes(
            symbol=cfg.symbol,
            starting_price=cfg.starting_price,
            steps=cfg.steps,
            step_seconds=cfg.step_seconds,
            volatility_annual=cfg.volatility_annual,
            spread_cents=cfg.spread_cents,
        )

        equity_curve: List[float] = []
        timestamps: List[datetime] = []
        max_equity = cfg.starting_cash
        max_drawdown = 0.0

        for q in quotes:
            # Mid price from quote
            if q.bid is not None and q.ask is not None:
                mid = (q.bid + q.ask) / 2.0
            elif q.last is not None:
                mid = float(q.last)
            else:
                # No usable price -> skip this step
                continue

            # Positions for risk; ExecutionTracker.summary() returns PositionState map
            positions_for_risk = self.tracker.summary()

            # Equity & snapshot BEFORE new orders
            cash_before = self._recompute_cash()
            equity_before = self._equity(mid, cash_before)
            snapshot = self._make_snapshot(equity_before, cash_before)

            # Run all strategies on this quote
            for strat in self.strategies:
                intents = strat.on_quote(q)
                for oi in intents:
                    self.order_manager.process_order(
                        oi,
                        q,
                        snapshot,
                        positions_for_risk,
                    )

            # AFTER orders: recompute cash + equity at same mid
            cash_after = self._recompute_cash()
            equity_after = self._equity(mid, cash_after)

            equity_curve.append(equity_after)
            if isinstance(q.timestamp, str):
                timestamps.append(datetime.fromisoformat(q.timestamp))
            else:
                timestamps.append(datetime.utcnow())

            if equity_after > max_equity:
                max_equity = equity_after
            if max_equity > 0:
                dd = (max_equity - equity_after) / max_equity
                if dd > max_drawdown:
                    max_drawdown = dd

        realized_pnl = sum(pos.realized_pnl for pos in self.tracker.positions.values())
        final_cash = self._recompute_cash()
        final_equity = equity_curve[-1] if equity_curve else cfg.starting_cash

        return BacktestResult(
            symbol=cfg.symbol.upper(),
            starting_cash=cfg.starting_cash,
            final_cash=final_cash,
            final_equity=final_equity,
            realized_pnl=realized_pnl,
            max_drawdown=max_drawdown,
            equity_curve=equity_curve,
            timestamps=timestamps,
            positions_summary=self.tracker.positions.copy(),
        )
