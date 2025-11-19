# HFTA/market/quote_provider.py

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from HFTA.broker.client import WealthsimpleClient, Quote

logger = logging.getLogger(__name__)

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yf = None  # type: ignore


class BaseQuoteProvider:
    """Abstract interface for quote providers.

    Implementations must return a mapping:
        { "AAPL": Quote(...), "MSFT": Quote(...), ... }
    for the requested symbols. Missing symbols can be omitted.
    """

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        raise NotImplementedError


class WealthsimpleQuoteProvider(BaseQuoteProvider):
    """Quote provider that wraps WealthsimpleClient.get_quote.

    Keeps backwards compatibility but adds optional parallelism so multiple
    symbols can be fetched at once.
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

        # Single-threaded fast path
        if len(symbols) == 1 or self.max_workers <= 1:
            for sym in symbols:
                q = self._fetch_one(sym)
                if q is not None:
                    quotes[sym] = q
            return quotes

        # Parallel fetch for multiple symbols
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fut_to_sym = {executor.submit(self._fetch_one, sym): sym for sym in symbols}
            for fut in as_completed(fut_to_sym):
                sym = fut_to_sym[fut]
                try:
                    q = fut.result()
                except Exception:
                    logger.exception(
                        "WealthsimpleQuoteProvider: error fetching %s", sym
                    )
                    continue
                if q is not None:
                    quotes[sym] = q

        return quotes


class FinnhubQuoteProvider(BaseQuoteProvider):
    """Quote provider using Finnhub's official API.

    - Uses /quote endpoint for near real-time prices.
    - Supports parallel fetch for multiple symbols, but enforces a per-minute
      call budget to avoid 429 Too Many Requests.

    API key resolution order:
      1) Explicit api_key argument
      2) Environment variable HFTA_FINNHUB_API_KEY
      3) Environment variable FINNHUB_API_KEY
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_workers: int = 4,
        timeout: float = 1.5,
        poll_interval: float = 1.0,
        max_calls_per_minute: int = 60,
        rate_limit_cooldown: float = 60.0,
    ) -> None:
        key = api_key or os.getenv("HFTA_FINNHUB_API_KEY") or os.getenv(
            "FINNHUB_API_KEY"
        )
        if not key:
            raise RuntimeError(
                "FinnhubQuoteProvider: API key is required. "
                "Set HFTA_FINNHUB_API_KEY or FINNHUB_API_KEY, "
                "or pass api_key explicitly."
            )
        self.api_key = key
        self.timeout = timeout
        self._session = requests.Session()

        # Rate limiting config
        self.max_calls_per_minute = max_calls_per_minute
        self.poll_interval = max(poll_interval, 0.001)
        self.rate_limit_cooldown = max(rate_limit_cooldown, 1.0)

        # Derived: how many symbols can we safely hit per loop?
        loops_per_minute = max(1, int(round(60.0 / self.poll_interval)))
        self._max_symbols_per_loop = max(
            1, self.max_calls_per_minute // loops_per_minute
        )

        # Don't spawn more workers than symbols per loop.
        self.max_workers = max(1, min(max_workers, self._max_symbols_per_loop))

        # State used when we hit 429
        self._rate_limited_until: float = 0.0
        self._last_rate_limit_log: float = 0.0

        logger.info(
            "FinnhubQuoteProvider: max_calls_per_minute=%d, poll_interval=%.3fs, "
            "loops_per_minute=%d, max_symbols_per_loop=%d, max_workers=%d",
            self.max_calls_per_minute,
            self.poll_interval,
            loops_per_minute,
            self._max_symbols_per_loop,
            self.max_workers,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _handle_rate_limit(self) -> None:
        now = time.time()
        self._rate_limited_until = now + self.rate_limit_cooldown
        # Log at most once every 10 seconds to avoid log spam
        if now - self._last_rate_limit_log > 10.0:
            self._last_rate_limit_log = now
            logger.warning(
                "FinnhubQuoteProvider: received 429 Too Many Requests. "
                "Pausing Finnhub calls for %.1f seconds.",
                self.rate_limit_cooldown,
            )

    def _fetch_one(self, symbol: str) -> Optional[Quote]:
        sym = symbol.upper()
        try:
            resp = self._session.get(
                f"{self.BASE_URL}/quote",
                params={"symbol": sym, "token": self.api_key},
                timeout=self.timeout,
            )
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                # Explicitly handle rate limiting
                if resp.status_code == 429:
                    logger.error(
                        "FinnhubQuoteProvider: rate limited while fetching %s: %s",
                        sym,
                        exc,
                    )
                    self._handle_rate_limit()
                    return None
                raise

            data = resp.json()

            # Finnhub /quote returns:
            #   c: current price
            #   h: high of day
            #   l: low of day
            #   o: open of day
            #   pc: previous close
            last = data.get("c")
            if last is None or last == 0:
                logger.debug(
                    "FinnhubQuoteProvider: no usable last price for %s: %s",
                    sym,
                    data,
                )
                return None

            last_f = float(last)
            now_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

            # Re-use HFTA.broker.client.Quote for downstream compatibility.
            return Quote(
                symbol=sym,
                security_id=sym,  # data-only; Wealthsimple still used for orders
                bid=last_f,
                ask=last_f,
                last=last_f,
                bid_size=None,
                ask_size=None,
                timestamp=now_iso,
            )
        except Exception:
            logger.exception("FinnhubQuoteProvider: failed to fetch quote for %s", sym)
            return None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        now = time.time()

        # If we're currently rate-limited, skip Finnhub calls this loop
        if now < self._rate_limited_until:
            logger.debug(
                "FinnhubQuoteProvider: currently rate limited until %.0f; "
                "skipping quote fetch for this loop.",
                self._rate_limited_until,
            )
            return {}

        symbols = [s.upper() for s in symbols]
        quotes: Dict[str, Quote] = {}

        if not symbols:
            return quotes

        # Enforce per-loop symbol budget
        if len(symbols) > self._max_symbols_per_loop:
            symbols = symbols[: self._max_symbols_per_loop]
            logger.debug(
                "FinnhubQuoteProvider: limiting this loop to %d symbols "
                "(configured max_symbols_per_loop).",
                self._max_symbols_per_loop,
            )

        # Single-threaded fast path
        if len(symbols) == 1 or self.max_workers <= 1:
            for sym in symbols:
                q = self._fetch_one(sym)
                if q is not None:
                    quotes[sym] = q
            return quotes

        # Parallel fetch for multiple symbols (within budget)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            fut_to_sym = {executor.submit(self._fetch_one, sym): sym for sym in symbols}
            for fut in as_completed(fut_to_sym):
                sym = fut_to_sym[fut]
                try:
                    q = fut.result()
                except Exception:
                    logger.exception(
                        "FinnhubQuoteProvider: error fetching %s", sym
                    )
                    continue
                if q is not None:
                    quotes[sym] = q

        return quotes


class YFinanceQuoteProvider(BaseQuoteProvider):
    """Quote provider using yfinance (Yahoo Finance) in a batched fashion.

    For multiple symbols, we call `yf.download` once with all tickers and
    use the latest 1-minute close for each symbol. This is suitable for
    development / DRY-RUN intraday testing but is not a low-latency
    production data feed.

    Requirements:
      - pip install yfinance
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

        # Import pandas lazily (yfinance depends on it)
        try:
            import pandas as pd  # type: ignore
        except Exception:
            logger.error(
                "YFinanceQuoteProvider: pandas is required but not available."
            )
            return quotes

        # Single-symbol fast path
        if len(symbols) == 1:
            sym = symbols[0]
            try:
                data = yf.download(
                    tickers=sym,
                    period="1d",
                    interval="1m",
                    auto_adjust=False,
                    progress=False,
                    group_by="ticker",
                    threads=False,
                )
            except Exception:
                logger.exception(
                    "YFinanceQuoteProvider: download failed for %s", sym
                )
                return quotes

            if data is None or data.empty:
                return quotes

            closes = data.get("Close")
            if closes is None or closes.dropna().empty:
                return quotes

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
            return quotes

        # Multi-symbol: single batch download
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
            logger.exception(
                "YFinanceQuoteProvider: batch download failed for %s", symbols
            )
            return quotes

        if data is None or data.empty:
            return quotes

        # When multiple tickers are requested, yfinance returns a DataFrame
        # with a MultiIndex on columns: (field, ticker) or (ticker, field)
        # depending on version. We handle both.
        if isinstance(data.columns, pd.MultiIndex):
            # Normalize to (ticker, field)
            level0 = [str(x) for x in data.columns.levels[0]]
            level1 = [str(x) for x in data.columns.levels[1]]

            fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
            if any(f in level0 for f in fields) and not any(
                f in level1 for f in fields
            ):
                # Columns like ('Close', 'AAPL') -> swap levels
                df = data.copy()
                df.columns = df.columns.swaplevel(0, 1)
            else:
                df = data

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
            # Fallback: assume data corresponds to the first symbol
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
