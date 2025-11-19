# HFTA/market/quote_provider.py

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    import yfinance as yf  # type: ignore
except Exception:  # yfinance is optional at runtime
    yf = None  # type: ignore

from HFTA.broker.client import WealthsimpleClient, Quote

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Base abstraction
# --------------------------------------------------------------------------- #


class BaseQuoteProvider(ABC):
    """Abstract base class for quote providers.

    Implementations must provide `get_quotes(symbols)` and return a mapping
    of UPPERCASED symbol -> Quote.
    """

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Wealthsimple-backed provider
# --------------------------------------------------------------------------- #


class WealthsimpleQuoteProvider(BaseQuoteProvider):
    """Quote provider that fetches prices directly from Wealthsimple.

    This is the most realistic provider for live / DRY-RUN trading since it
    uses the same data source as order routing.
    """

    def __init__(self, client: WealthsimpleClient, max_workers: int = 4) -> None:
        self.client = client
        self.max_workers = max_workers

    def _fetch_one(self, symbol: str) -> Optional[Quote]:
        try:
            return self.client.get_quote(symbol)
        except Exception:
            logger.exception(
                "WealthsimpleQuoteProvider: failed to fetch quote for %s", symbol
            )
            return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        symbols = [s.upper() for s in symbols]
        quotes: Dict[str, Quote] = {}
        if not symbols:
            return quotes

        # Single-threaded path for small batches or 1 symbol
        if len(symbols) == 1 or self.max_workers <= 1:
            for sym in symbols:
                q = self._fetch_one(sym)
                if q is not None:
                    quotes[sym] = q
            return quotes

        # Multi-threaded for larger batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(self._fetch_one, sym): sym for sym in symbols}
            for fut in as_completed(future_map):
                sym = future_map[fut]
                try:
                    q = fut.result()
                except Exception:
                    logger.exception(
                        "WealthsimpleQuoteProvider: worker failed for %s", sym
                    )
                    continue
                if q is not None:
                    quotes[sym] = q

        return quotes


# --------------------------------------------------------------------------- #
# Finnhub-backed provider
# --------------------------------------------------------------------------- #


class FinnhubQuoteProvider(BaseQuoteProvider):
    """Quote provider using Finnhub's `/quote` endpoint."""

    BASE_URL = "https://finnhub.io/api/v1/quote"

    def __init__(
        self,
        api_key: str,
        max_workers: int = 4,
        timeout: float = 1.5,
        poll_interval: float = 1.0,
        max_calls_per_minute: int = 60,
        cooldown_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("FinnhubQuoteProvider: api_key is required")

        self.api_key = api_key
        self.max_workers = max_workers
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_calls_per_minute = max_calls_per_minute
        self.cooldown_seconds = cooldown_seconds

        self._session = requests.Session()
        self._window_start = time.time()
        self._calls_in_window = 0

    # --- rate limit helpers ------------------------------------------------ #

    def _allow_call(self, n_symbols: int) -> bool:
        """Very simple per-minute call budget."""
        now = time.time()
        elapsed = now - self._window_start
        if elapsed >= 60:
            self._window_start = now
            self._calls_in_window = 0

        projected = self._calls_in_window + n_symbols
        if projected > self.max_calls_per_minute:
            logger.warning(
                "FinnhubQuoteProvider: rate limit would be exceeded "
                "(%d calls in window, max=%d); skipping this batch.",
                self._calls_in_window,
                self.max_calls_per_minute,
            )
            return False

        self._calls_in_window = projected
        return True

    def _fetch_one(self, symbol: str) -> Optional[Quote]:
        params = {"symbol": symbol, "token": self.api_key}
        try:
            resp = self._session.get(self.BASE_URL, params=params, timeout=self.timeout)
        except Exception:
            logger.exception("FinnhubQuoteProvider: request failed for %s", symbol)
            return None

        if resp.status_code == 429:
            logger.warning(
                "FinnhubQuoteProvider: HTTP 429 for %s (rate limited)", symbol
            )
            return None
        if not resp.ok:
            logger.warning(
                "FinnhubQuoteProvider: non-200 response for %s: %s %s",
                symbol,
                resp.status_code,
                resp.text[:200],
            )
            return None

        try:
            data = resp.json()
        except Exception:
            logger.exception("FinnhubQuoteProvider: JSON decode failed for %s", symbol)
            return None

        last = data.get("c")
        bid = data.get("b")
        ask = data.get("a")

        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        last_f = _to_float(last)
        bid_f = _to_float(bid)
        ask_f = _to_float(ask)

        if all(v is None for v in (last_f, bid_f, ask_f)):
            logger.debug(
                "FinnhubQuoteProvider: no usable prices for %s: %r", symbol, data
            )
            return None

        px = last_f or bid_f or ask_f
        if px is None:
            return None

        now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

        return Quote(
            symbol=symbol,
            security_id=symbol,
            bid=bid_f or px,
            ask=ask_f or px,
            last=last_f or px,
            bid_size=None,
            ask_size=None,
            timestamp=now_iso,
        )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        symbols = [s.upper() for s in symbols]
        quotes: Dict[str, Quote] = {}
        if not symbols:
            return quotes

        if not self._allow_call(len(symbols)):
            # Rate limit hit: skip this batch.
            return quotes

        if len(symbols) == 1 or self.max_workers <= 1:
            for sym in symbols:
                q = self._fetch_one(sym)
                if q is not None:
                    quotes[sym] = q
            return quotes

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {executor.submit(self._fetch_one, sym): sym for sym in symbols}
            for fut in as_completed(future_map):
                sym = future_map[fut]
                try:
                    q = fut.result()
                except Exception:
                    logger.exception("FinnhubQuoteProvider: worker failed for %s", sym)
                    continue
                if q is not None:
                    quotes[sym] = q

        return quotes


# --------------------------------------------------------------------------- #
# yfinance-backed provider (batched)
# --------------------------------------------------------------------------- #


class YFinanceQuoteProvider(BaseQuoteProvider):
    """Quote provider using yfinance (Yahoo Finance) with batched fetching.

    This is mainly for DRY-RUN / research. It uses `yf.download` once per batch
    of symbols (even if there is only one symbol), and extracts the latest
    close as a proxy for bid/ask/last.
    """

    def __init__(self) -> None:
        if yf is None:
            raise RuntimeError(
                "YFinanceQuoteProvider: yfinance is not installed. "
                "Install with: pip install yfinance"
            )

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        symbols = [s.upper() for s in symbols]
        quotes: Dict[str, Quote] = {}
        if not symbols:
            return quotes

        try:
            import pandas as pd  # type: ignore
        except Exception:
            logger.error("YFinanceQuoteProvider: pandas is required but not available.")
            return quotes

        tickers_str = " ".join(symbols)
        try:
            data = yf.download(
                tickers=tickers_str,
                period="1d",
                interval="1m",
                auto_adjust=False,
                progress=False,
                group_by="ticker",
                threads=False,
            )
        except Exception:
            logger.exception("YFinanceQuoteProvider: download failed for %s", symbols)
            return quotes

        if data is None or data.empty:
            return quotes

        # MultiIndex columns case: (ticker, field)
        if isinstance(data.columns, pd.MultiIndex):
            df = data.copy()

            fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            level0 = [str(x) for x in df.columns.levels[0]]
            level1 = [str(x) for x in df.columns.levels[1]]

            # If fields are on level 0 instead of level 1, swap levels.
            if any(f in level0 for f in fields) and not any(f in level1 for f in fields):
                df.columns = df.columns.swaplevel(0, 1)

            for sym in symbols:
                if sym not in df.columns.levels[0]:
                    continue
                sub = df[sym]
                closes = sub.get("Close")
                if closes is None or closes.dropna().empty:
                    continue
                last = float(closes.dropna().iloc[-1])
                ts = closes.index[-1].to_pydatetime().isoformat()
                quotes[sym] = Quote(
                    symbol=sym,
                    security_id=sym,
                    bid=last,
                    ask=last,
                    last=last,
                    bid_size=None,
                    ask_size=None,
                    timestamp=ts,
                )
        else:
            # Single-index DataFrame (usually when there is a single ticker)
            closes = data.get("Close")
            if closes is not None and not closes.dropna().empty:
                last = float(closes.dropna().iloc[-1])
                ts = closes.index[-1].to_pydatetime().isoformat()
                sym = symbols[0]
                quotes[sym] = Quote(
                    symbol=sym,
                    security_id=sym,
                    bid=last,
                    ask=last,
                    last=last,
                    bid_size=None,
                    ask_size=None,
                    timestamp=ts,
                )

        return quotes
