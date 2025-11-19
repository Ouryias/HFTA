# HFTA/core/risk_manager.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, Mapping, Tuple

from HFTA.broker.client import Quote, PortfolioSnapshot, Holding
from HFTA.strategies.base import OrderIntent

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """
    Very simple risk configuration.

    Per-order checks
    ----------------
    max_notional_per_order:
        Absolute cap per order in account currency
        (e.g. 100.0 = 100 CAD/USD).
    max_cash_utilization:
        Fraction of current available cash that a single BUY can use
        (e.g. 0.1 = 10% of available cash).
    allow_short_selling:
        If False, SELL quantity may not exceed current long position
        (no opening new shorts).

    Portfolio-level checks (optional)
    ---------------------------------
    max_total_exposure_ratio:
        Optional cap on total absolute exposure relative to account net worth.
        Example: 1.5 means sum(|position_qty * price|) must not exceed
        150% of snapshot.net_worth. If None, this check is disabled.
    max_positions:
        Optional cap on the number of open positions (symbols with
        non-zero quantity). If None, this check is disabled.
    """
    max_notional_per_order: float = 100.0
    max_cash_utilization: float = 0.1
    allow_short_selling: bool = False
    max_total_exposure_ratio: Optional[float] = None
    max_positions: Optional[int] = None


class RiskManager:
    """
    Stateless per-order risk checks.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ #
    # Basic helpers
    # ------------------------------------------------------------------ #

    def _infer_price(self, oi: OrderIntent, quote: Quote) -> Optional[float]:
        """
        Pick a reasonable execution price for risk checks.

        Preference order:
            1) Order limit price
            2) Quote last
            3) For BUY  -> quote.ask
               For SELL -> quote.bid
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

    def _holding_qty(self, symbol: str, positions: Mapping[str, Any]) -> float:
        """
        Extract position quantity for `symbol` from either a Holding or
        PositionState-like object.
        """
        h: Any = positions.get(symbol.upper())
        if h is None:
            return 0.0
        qty = getattr(h, "quantity", None)
        if qty is None:
            return 0.0
        try:
            return float(qty)
        except (TypeError, ValueError):
            return 0.0

    def _holding_price(self, symbol: str, positions: Mapping[str, Any]) -> Optional[float]:
        """
        Best effort estimate of the price associated with a position.
        Used only for portfolio-level exposure estimation.
        """
        h: Any = positions.get(symbol.upper())
        if h is None:
            return None
        price = getattr(h, "avg_price", None)
        try:
            return float(price) if price is not None else None
        except (TypeError, ValueError):
            return None

    def _portfolio_exposure(
        self,
        snapshot: PortfolioSnapshot,
        positions: Mapping[str, Any],
        current_symbol: str,
        current_side: str,
        current_qty: float,
        current_price: float,
    ) -> Tuple[float, int]:
        """
        Estimate portfolio exposure AFTER applying the proposed order.

        Returns:
            (total_exposure, open_positions_count)
        """
        symbol_u = current_symbol.upper()
        side = current_side.lower()

        # First compute exposure and open positions from existing holdings.
        total_exposure = 0.0
        open_positions = 0
        for sym, pos in positions.items():
            qty = getattr(pos, "quantity", 0.0)
            try:
                qty_f = float(qty)
            except (TypeError, ValueError):
                qty_f = 0.0
            if qty_f == 0.0:
                continue
            price = getattr(pos, "avg_price", None)
            try:
                price_f = float(price) if price is not None else 0.0
            except (TypeError, ValueError):
                price_f = 0.0
            if price_f <= 0.0:
                # Fallback: use snapshot net_worth / 100 to avoid zero
                price_f = max(float(snapshot.net_worth or 0.0) / 100.0, 0.01)
            total_exposure += abs(qty_f * price_f)
            open_positions += 1

        # Adjust for the proposed order on the current symbol.
        existing_qty = self._holding_qty(symbol_u, positions)
        new_qty = existing_qty
        if side == "buy":
            new_qty += current_qty
        elif side == "sell":
            new_qty -= current_qty

        # Exposure contribution from this symbol BEFORE the new order.
        existing_price = self._holding_price(symbol_u, positions) or current_price
        existing_exposure = abs(existing_qty * existing_price)
        new_exposure = abs(new_qty * current_price)

        total_exposure = total_exposure - existing_exposure + new_exposure

        # Adjust open positions count.
        if existing_qty == 0.0 and new_qty != 0.0:
            open_positions += 1
        elif existing_qty != 0.0 and new_qty == 0.0:
            open_positions = max(open_positions - 1, 0)

        return total_exposure, open_positions

    # ------------------------------------------------------------------ #
    # Main API
    # ------------------------------------------------------------------ #

    def approve(
        self,
        oi: OrderIntent,
        quote: Quote,
        snapshot: PortfolioSnapshot,
        positions: Dict[str, Any],
    ) -> bool:
        """
        Return True if the order is allowed under the configured risk limits.
        """
        price = self._infer_price(oi, quote)
        if price is None:
            logger.info("Risk: rejecting %s (no usable price)", oi)
            return False

        notional = price * oi.quantity
        side = oi.side.lower()

        # 1) Hard cap per order
        if notional > self.config.max_notional_per_order:
            logger.info(
                "Risk: rejecting %s (notional %.2f > max_notional_per_order %.2f)",
                oi,
                notional,
                self.config.max_notional_per_order,
            )
            return False

        # 2) Per-order cash utilization for BUYs
        if side == "buy":
            cash_avail = float(snapshot.cash_available or 0.0)
            max_allowed = cash_avail * self.config.max_cash_utilization
            if notional > max_allowed:
                logger.info(
                    "Risk: rejecting %s (notional %.2f > cash_allowed %.2f)",
                    oi,
                    notional,
                    max_allowed,
                )
                return False

        # 3) SELLs: prevent opening shorts unless allowed
        if side == "sell" and not self.config.allow_short_selling:
            held_qty = self._holding_qty(oi.symbol, positions)
            if held_qty <= 0 or oi.quantity > held_qty:
                logger.info(
                    "Risk: rejecting %s (sell qty %.2f > holdings %.2f)",
                    oi,
                    oi.quantity,
                    held_qty,
                )
                return False

        # 4) Portfolio-level checks (optional)
        if (
            self.config.max_total_exposure_ratio is not None
            or self.config.max_positions is not None
        ):
            total_exposure, open_positions = self._portfolio_exposure(
                snapshot=snapshot,
                positions=positions,
                current_symbol=oi.symbol,
                current_side=oi.side,
                current_qty=oi.quantity,
                current_price=price,
            )

            net_worth = float(snapshot.net_worth or 0.0)
            if (
                self.config.max_total_exposure_ratio is not None
                and net_worth > 0.0
            ):
                exposure_ratio = total_exposure / net_worth
                if exposure_ratio > self.config.max_total_exposure_ratio:
                    logger.info(
                        "Risk: rejecting %s (exposure_ratio %.3f > max_total_exposure_ratio %.3f)",
                        oi,
                        exposure_ratio,
                        self.config.max_total_exposure_ratio,
                    )
                    return False

            if (
                self.config.max_positions is not None
                and open_positions > self.config.max_positions
            ):
                logger.info(
                    "Risk: rejecting %s (open_positions %d > max_positions %d)",
                    oi,
                    open_positions,
                    self.config.max_positions,
                )
                return False

        return True
