# scripts/run_engine.py

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from HFTA.broker.client import WealthsimpleClient
from HFTA.core.engine import Engine
from HFTA.core.execution_tracker import ExecutionTracker
from HFTA.core.order_manager import OrderManager
from HFTA.core.risk_manager import RiskManager
from HFTA.logging_utils import setup_logging, parse_log_level
from HFTA.market.quote_provider import (
    WealthsimpleQuoteProvider,
    FinnhubQuoteProvider,
    YFinanceQuoteProvider,
)
from HFTA.market.intraday_stats import IntradayStatsTracker
from HFTA.market.universe import MarketUniverseConfig, MarketUniverse
from HFTA.ai.controller import AIController
from HFTA.symbol_selection import SymbolSelector
from HFTA.config_loader import load_config

logger = logging.getLogger(__name__)


def build_quote_provider(data_block, client: WealthsimpleClient, poll_interval: float):
    source = (data_block.get("quote_source") or "wealthsimple").lower()
    max_workers = int(data_block.get("max_workers", 4))

    if source == "wealthsimple":
        logger.info("Using WealthsimpleQuoteProvider for quotes")
        return WealthsimpleQuoteProvider(client=client, max_workers=max_workers)

    if source == "finnhub":
        api_key = data_block.get("finnhub_api_key")
        if not api_key:
            raise ValueError("data.finnhub_api_key is required when quote_source='finnhub'")
        max_calls_per_minute = int(data_block.get("finnhub_max_calls_per_minute", 60))
        cooldown = float(data_block.get("finnhub_rate_limit_cooldown", 60.0))
        logger.info(
            "Using FinnhubQuoteProvider for quotes (max_workers=%d, max_calls_per_minute=%d)",
            max_workers,
            max_calls_per_minute,
        )
        return FinnhubQuoteProvider(
            api_key=api_key,
            max_workers=max_workers,
            timeout=1.5,
            poll_interval=poll_interval,
            max_calls_per_minute=max_calls_per_minute,
            cooldown_seconds=cooldown,
        )

    if source == "yfinance":
        logger.info("Using YFinanceQuoteProvider for quotes (batched)")
        return YFinanceQuoteProvider()

    raise ValueError(f"Unknown quote_source={source!r}")


def maybe_build_universe(universe_block) -> list[str] | None:
    """
    Build a dynamic MarketUniverse and return its symbol list, or None.

    This closely follows the original implementation:

    - Reads parameters from the `universe` block in JSON.
    - Reads the Polygon API key from environment:
        HFTA_POLYGON_API_KEY or POLYGON_API_KEY.
    - If no API key is present, logs a warning and returns None.
    - On success, calls universe.refresh() once and returns the symbols.
    """
    if not universe_block or not universe_block.get("enabled", False):
        logger.info("MarketUniverse disabled in config; using static symbols list.")
        return None

    max_symbols = int(universe_block.get("max_symbols", 50))
    min_price = float(universe_block.get("min_price", 5.0))
    max_price = float(universe_block.get("max_price", 500.0))
    min_dv = float(universe_block.get("min_dollar_volume", 20_000_000.0))
    lookback_days = int(universe_block.get("lookback_days", 3))

    cfg_obj = MarketUniverseConfig(
        max_symbols=max_symbols,
        min_price=min_price,
        max_price=max_price,
        min_dollar_volume=min_dv,
        lookback_days=lookback_days,
    )

    api_key = os.getenv("HFTA_POLYGON_API_KEY") or os.getenv("POLYGON_API_KEY")
    if not api_key:
        logger.warning(
            "MarketUniverse enabled but no Polygon API key found "
            "(HFTA_POLYGON_API_KEY / POLYGON_API_KEY). Universe will not be built."
        )
        return None

    universe = MarketUniverse(config=cfg_obj, api_key=api_key)
    logger.info(
        "MarketUniverse created (max_symbols=%d, min_price=%.2f, max_price=%.2f, "
        "min_dollar_volume=%.0f, lookback_days=%d)",
        cfg_obj.max_symbols,
        cfg_obj.min_price,
        cfg_obj.max_price,
        cfg_obj.min_dollar_volume,
        cfg_obj.lookback_days,
    )

    try:
        universe.refresh()
    except Exception as exc:
        logger.exception(
            "MarketUniverse: failed to refresh universe; falling back to static symbols: %s",
            exc,
        )
        return None

    if not universe.symbols:
        logger.warning(
            "MarketUniverse.refresh completed but returned an empty symbol list; "
            "falling back to static symbols."
        )
        return None

    return [s.upper() for s in universe.symbols]


def maybe_build_ai_controller(ai_block) -> AIController | None:
    if not ai_block or not ai_block.get("enabled", False):
        return None
    return AIController(
        model=ai_block.get("model", "gpt-5-mini"),
        interval_loops=int(ai_block.get("interval_loops", 60))
    )


def maybe_build_symbol_selector(symbol_selector_block) -> SymbolSelector | None:
    if not symbol_selector_block or not symbol_selector_block.get("enabled", False):
        return None
    return SymbolSelector(
        interval_loops=int(symbol_selector_block.get("interval_loops", 60)),
        min_trades=int(symbol_selector_block.get("min_trades", 3)),
        mode=symbol_selector_block.get("mode", "heuristic"),
        model=symbol_selector_block.get("model", "gpt-5-mini"),
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run HFTA live / DRY-RUN engine.")
    parser.add_argument(
        "--config",
        default="configs/paper_aapl.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--log-file",
        default="logs/engine.log",
        help="Path to log file.",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        help="Log level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args(argv)

    log_level = parse_log_level(args.log_level)
    setup_logging("HFTA.engine", args.log_file, level=log_level)

    # Turn down Peewee spam
    logging.getLogger("peewee").setLevel(logging.WARNING)

    logger.info("Starting engine with config=%s", args.config)

    loaded = load_config(args.config)

    # Universe (optional) can override symbols
    universe_symbols = maybe_build_universe(loaded.universe_block)
    if universe_symbols:
        symbols = universe_symbols
        logger.info(
            "Using dynamic universe from Polygon: %d symbols (first few: %s)",
            len(symbols),
            symbols[:10],
        )
    else:
        symbols = loaded.symbols
        logger.info("Using static symbols from config: %s", symbols)

    client = WealthsimpleClient()
    tracker = ExecutionTracker()
    risk_manager = RiskManager(config=loaded.risk_config)
    order_manager = OrderManager(
        client=client,
        risk_manager=risk_manager,
        execution_tracker=tracker,
        live=False,  # DRY-RUN for now
    )

    quote_provider = build_quote_provider(loaded.data_block, client=client, poll_interval=loaded.poll_interval)
    intraday_stats = IntradayStatsTracker()
    ai_controller = maybe_build_ai_controller(loaded.ai_block)
    symbol_selector = maybe_build_symbol_selector(loaded.symbol_selector_block)

    engine = Engine(
        client=client,
        strategies=loaded.strategies,
        symbols=symbols,
        order_manager=order_manager,
        quote_provider=quote_provider,
        poll_interval=loaded.poll_interval,
        paper_cash=loaded.paper_cash,
        ai_controller=ai_controller,
        intraday_stats=intraday_stats,
        symbol_selector=symbol_selector,
    )

    try:
        engine.run_forever()
    except Exception:
        logger.exception("Engine terminated due to an unhandled exception")
        raise


if __name__ == "__main__":
    main()
