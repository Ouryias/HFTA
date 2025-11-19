# HFTA/market/quote_recorder.py

from __future__ import annotations

from datetime import datetime
from typing import Optional

from HFTA.broker.client import Quote as WSQuote
from HFTA.market.db import db
from HFTA.market.models import QuoteBar


def record_quote(symbol: str, quote: WSQuote, source: str = "wealthsimple") -> None:
    """
    Persist a single quote snapshot to Postgres.

    Call this from the engine loop for each symbol when a quote is available.
    """
    # Defensive: if everything is None, skip
    if quote.last is None and quote.bid is None and quote.ask is None:
        return

    # Use quote.timestamp if it's usable, else now()
    ts_value: datetime
    if getattr(quote, "timestamp", None):
        ts_value = datetime.utcnow()
    else:
        ts_value = datetime.utcnow()

    with db.atomic():
        QuoteBar.create(
            symbol=symbol.upper(),
            ts=ts_value,
            last=quote.last,
            bid=quote.bid,
            ask=quote.ask,
            source=source,
        )
