# HFTA/broker/client.py

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from HFTA.wealthsimple_v2 import WealthsimpleV2

logger = logging.getLogger(__name__)


@dataclass
class Quote:
    symbol: str
    security_id: str
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    bid_size: Optional[float]
    ask_size: Optional[float]
    timestamp: Optional[str]


@dataclass
class PortfolioSnapshot:
    account_id: str
    currency: str
    net_worth: float
    cash_available: float


class WealthsimpleClient:
    """
    Thin wrapper around WealthsimpleV2 for the HFTA engine.
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        currency: str = "CAD",
        ws: Optional[WealthsimpleV2] = None,
    ) -> None:
        self.ws = ws or WealthsimpleV2()
        self.currency = currency
        self._security_cache: Dict[str, str] = {}
        self._account_id = account_id or self._auto_pick_default_account()
        logger.info("WealthsimpleClient initialized for account %s", self._account_id)

    # ------------------------------------------------------------------ #
    # Account helpers
    # ------------------------------------------------------------------ #

    def _auto_pick_default_account(self) -> str:
        accounts = self.ws.get_accounts()
        if not accounts:
            raise RuntimeError("No accounts returned from Wealthsimple API.")
        for acc in accounts:
            if acc.get("status") == "OPEN":
                return acc["id"]
        return accounts[0]["id"]

    @property
    def account_id(self) -> str:
        return self._account_id

    # ------------------------------------------------------------------ #
    # Security resolution + quotes
    # ------------------------------------------------------------------ #

    def resolve_security_id(self, symbol: str, exchange: Optional[str] = None) -> str:
        key = symbol.upper() if not exchange else f"{symbol.upper()}:{exchange}"
        if key in self._security_cache:
            return self._security_cache[key]

        try:
            sec_id = self.ws.get_ticker_id(symbol, exchange=exchange)
        except Exception:
            results = self.ws.search_securities(symbol)
            cand_id = None
            for r in results:
                stock = r.get("stock", {})
                if stock.get("symbol", "").upper() == symbol.upper():
                    cand_id = r["id"]
                    break
            if not cand_id and results:
                cand_id = results[0]["id"]
            if not cand_id:
                raise ValueError(f"Could not resolve security_id for {symbol}")
            sec_id = cand_id

        self._security_cache[key] = sec_id
        return sec_id

    def get_quote(self, symbol: str, exchange: Optional[str] = None) -> Quote:
        sec_id = self.resolve_security_id(symbol, exchange)
        q = self.ws.get_security_quote(sec_id, currency=self.currency)

        return Quote(
            symbol=symbol.upper(),
            security_id=sec_id,
            bid=q.get("bid"),
            ask=q.get("ask"),
            last=q.get("price") or q.get("last"),
            bid_size=q.get("bidSize") or q.get("bid_size"),
            ask_size=q.get("askSize") or q.get("ask_size"),
            timestamp=q.get("timestamp"),
        )

    # ------------------------------------------------------------------ #
    # Portfolio snapshot
    # ------------------------------------------------------------------ #

    def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        fin = self.ws.get_account_financials([self._account_id], currency=self.currency)
        f = fin[0] if isinstance(fin, list) else fin["financials"][0]

        net_worth = f["netWorth"]["amount"]
        cash_available = f["buyingPower"]["amount"]

        return PortfolioSnapshot(
            account_id=self._account_id,
            currency=self.currency,
            net_worth=net_worth,
            cash_available=cash_available,
        )

    # ------------------------------------------------------------------ #
    # Equity orders (basic)
    # ------------------------------------------------------------------ #

    def place_equity_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        account_id = self._account_id
        sec_id = self.resolve_security_id(symbol)

        side = side.lower()
        order_type = order_type.lower()

        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")

        if order_type == "market":
            if side == "buy":
                return self.ws.market_buy(account_id, sec_id, quantity)
            else:
                return self.ws.market_sell(account_id, sec_id, quantity)

        if order_type == "limit":
            if limit_price is None:
                raise ValueError("limit_price required for limit")
            if side == "buy":
                return self.ws.limit_buy(account_id, sec_id, quantity, limit_price)
            else:
                return self.ws.limit_sell(account_id, sec_id, quantity, limit_price)

        raise ValueError(f"Unsupported order_type: {order_type}")
