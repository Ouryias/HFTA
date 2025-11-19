# scripts/run_engine.py

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from HFTA.ai.controller import AIController
from HFTA.broker.client import WealthsimpleClient
from HFTA.core.engine import Engine
from HFTA.core.execution_tracker import ExecutionTracker
from HFTA.core.order_manager import OrderManager
from HFTA.core.risk_manager import RiskConfig, RiskManager
from HFTA.logging_utils import setup_logging, parse_log_level
from HFTA.market.quote_provider import (
    BaseQuoteProvider,
    WealthsimpleQuoteProvider,
    FinnhubQuoteProvider,
    YFinanceQuoteProvider,
)
from HFTA.market.universe import (
    MarketUniverseConfig,
    MarketUniverse,
)
from HFTA.market.intraday_stats import IntradayStatsTracker
from HFTA.symbol_selection import SymbolSelector
from HFTA.strategies.micro_market_maker import MicroMarketMaker
from HFTA.strategies.micro_trend_scalper import MicroTrendScalper


# Map strategy type strings in the config to concrete classes
STRATEGY_REGISTRY: Dict[str, Any] = {
    "micro_market_maker": MicroMarketMaker,
    "micro_trend_scalper": MicroTrendScalper,
}


def load_config(path: Path) -> Dict[str, Any]:
    """Load a JSON config file from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_strategies(cfg: Dict[str, Any], logger) -> List[Any]:
    """Instantiate strategy objects from the config."""
    strategies_cfg = cfg.get("strategies", [])
    strategies: List[Any] = []

    for s in strategies_cfg:
        s_type = s["type"]
        name = s["name"]
        s_conf = s.get("config", {})

        cls = STRATEGY_REGISTRY.get(s_type)
        if cls is None:
            raise ValueError(f"Unknown strategy type in config: {s_type!r}")

        logger.debug(
            "Building strategy '%s' of type '%s' with config=%s",
            name,
            s_type,
            s_conf,
        )
        strategies.append(cls(name=name, config=s_conf))

    return strategies


def build_ai_controller(cfg: Dict[str, Any], logger) -> Optional[AIController]:
    """Create AIController if enabled in config."""
    ai_cfg = cfg.get("ai", {})
    enabled = bool(ai_cfg.get("enabled", False))
    if not enabled:
        logger.debug("AIController disabled in config.")
        return None

    model = ai_cfg.get("model", "gpt-5-mini")
    interval_loops = int(ai_cfg.get("interval_loops", 12))
    temperature = float(ai_cfg.get("temperature", 0.2))

    controller = AIController(
        model=model,
        interval_loops=interval_loops,
        temperature=temperature,
        enabled=True,
    )
    logger.debug(
        "AIController created: model=%s, interval_loops=%d, temperature=%.3f",
        controller.model,
        controller.interval_loops,
        controller.temperature,
    )
    return controller


def build_market_universe(cfg: Dict[str, Any], logger) -> Optional[MarketUniverse]:
    """Construct a dynamic MarketUniverse from config, if enabled."""
    u_cfg = cfg.get("universe", {})
    enabled = bool(u_cfg.get("enabled", False))
    if not enabled:
        logger.info("MarketUniverse disabled in config; using static symbols list.")
        return None

    max_symbols = int(u_cfg.get("max_symbols", 50))
    min_price = float(u_cfg.get("min_price", 5.0))
    max_price = float(u_cfg.get("max_price", 500.0))
    min_dv = float(u_cfg.get("min_dollar_volume", 20_000_000.0))
    lookback_days = int(u_cfg.get("lookback_days", 3))

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

    return universe


def build_quote_provider(
    cfg: Dict[str, Any],
    logger,
    client: WealthsimpleClient,
    poll_interval: float,
) -> BaseQuoteProvider:
    """Factory for the quote provider used by the engine.

    Config block (optional):

      "data": {
        "quote_source": "wealthsimple" | "finnhub" | "yfinance",
        "max_workers": 4,
        "finnhub_api_key": "...optional...",
        "finnhub_max_calls_per_minute": 60,
        "finnhub_rate_limit_cooldown": 60.0
      }

    - If "data" is missing, default is WealthsimpleQuoteProvider.
    - For production with a paid data plan, prefer "finnhub".
    - For development / intraday testing where you hit Finnhub limits,
      you can use "yfinance".
    """
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("quote_source", "wealthsimple").lower()
    max_workers = int(data_cfg.get("max_workers", 4))
    finnhub_api_key = data_cfg.get("finnhub_api_key")
    finnhub_max_calls_per_minute = int(
        data_cfg.get("finnhub_max_calls_per_minute", 60)
    )
    finnhub_rate_limit_cooldown = float(
        data_cfg.get("finnhub_rate_limit_cooldown", 60.0)
    )

    if source == "finnhub":
        logger.info(
            "Using FinnhubQuoteProvider for quotes (max_workers=%d, "
            "max_calls_per_minute=%d).",
            max_workers,
            finnhub_max_calls_per_minute,
        )
        provider = FinnhubQuoteProvider(
            api_key=finnhub_api_key,
            max_workers=max_workers,
            timeout=1.5,
            poll_interval=poll_interval,
            max_calls_per_minute=finnhub_max_calls_per_minute,
            rate_limit_cooldown=finnhub_rate_limit_cooldown,
        )
        return provider

    if source == "yfinance":
        logger.info("Using YFinanceQuoteProvider for quotes (batched).")
        provider = YFinanceQuoteProvider()
        return provider

    logger.info(
        "Using WealthsimpleQuoteProvider for quotes (max_workers=%d).", max_workers
    )
    return WealthsimpleQuoteProvider(client=client, max_workers=max_workers)


def build_symbol_selector(
    cfg: Dict[str, Any],
    logger,
    market_universe: Optional[MarketUniverse],
) -> Optional[SymbolSelector]:
    """Create SymbolSelector (dynamic symbol picker) if enabled in config.

    Config block (optional):

      "symbol_selector": {
        "enabled": true,
        "interval_loops": 60,
        "min_trades": 3,
        "mode": "hybrid",  // "heuristic" | "gpt" | "hybrid"
        "model": "gpt-5-mini"
      }
    """
    ss_cfg = cfg.get("symbol_selector", {})
    enabled = bool(ss_cfg.get("enabled", False))
    if not enabled:
        logger.info("SymbolSelector disabled in config.")
        return None

    interval_loops = int(ss_cfg.get("interval_loops", 60))
    min_trades = int(ss_cfg.get("min_trades", 3))
    mode = ss_cfg.get("mode", "hybrid")
    model = ss_cfg.get("model", "gpt-5-mini")

    selector = SymbolSelector(
        market_universe=market_universe,
        interval_loops=interval_loops,
        min_trades=min_trades,
        enabled=True,
        mode=mode,
        model=model,
    )

    logger.info(
        "SymbolSelector enabled: mode=%s, interval_loops=%d, min_trades=%d",
        selector.mode,
        selector.interval_loops,
        selector.min_trades,
    )
    return selector


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the HFTA engine in DRY-RUN mode."
    )
    parser.add_argument(
        "--config",
        default="configs/paper_aapl.json",
        help="Path to JSON config file (default: configs/paper_aapl.json)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/engine.log",
        help="Path to log file (default: logs/engine.log).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        help="Logging level: DEBUG, INFO, WARNING, ERROR (default: DEBUG).",
    )

    args = parser.parse_args()

    level = parse_log_level(args.log_level)
    logger = setup_logging(
        "HFTA.engine",
        log_file=args.log_file,
        level=level,
        log_to_console=True,
    )
    logger.debug("Parsed arguments: %s", vars(args))
    logger.info("Starting run_engine with config=%s", args.config)

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    logger.info("Loaded config from %s", cfg_path)
    logger.debug("Config JSON: %s", cfg)

    paper_cash = float(cfg.get("paper_cash", 0.0)) or None
    poll_interval = float(cfg.get("poll_interval", 5.0))

    # 1) Dynamic market universe (overall market)
    market_universe = build_market_universe(cfg, logger)

    # 2) Symbols: if universe exists, use it; otherwise fall back to static config.
    if market_universe is not None and market_universe.symbols:
        symbols = [s.upper() for s in market_universe.symbols]
        logger.info(
            "Using dynamic universe with %d symbols (ignoring static 'symbols' in config).",
            len(symbols),
        )
    else:
        symbols = [s.upper() for s in cfg.get("symbols", ["AAPL"])]
        logger.info("Using static symbol list from config: %s", symbols)

    # 3) Risk configuration
    risk_cfg_raw = cfg.get("risk", {})
    risk_cfg = RiskConfig(
        max_notional_per_order=float(
            risk_cfg_raw.get("max_notional_per_order", 1000.0)
        ),
        max_cash_utilization=float(
            risk_cfg_raw.get("max_cash_utilization", 0.10)
        ),
        allow_short_selling=bool(
            risk_cfg_raw.get("allow_short_selling", False)
        ),
    )
    risk_manager = RiskManager(risk_cfg)
    logger.info("RiskConfig: %s", risk_cfg)

    # 4) Broker + trackers
    client = WealthsimpleClient()
    execution_tracker = ExecutionTracker()
    order_manager = OrderManager(
        client=client,
        risk_manager=risk_manager,
        execution_tracker=execution_tracker,
        live=False,  # DRY-RUN; live mode comes later
    )

    # 5) Strategies, AI controller, quote provider, symbol selector
    strategies = build_strategies(cfg, logger)
    ai_controller = build_ai_controller(cfg, logger)
    quote_provider = build_quote_provider(cfg, logger, client, poll_interval)
    symbol_selector = build_symbol_selector(cfg, logger, market_universe)
    intraday_stats = IntradayStatsTracker() if symbol_selector is not None else None

    logger.info(
        "Built %d strategies for symbols=%s", len(strategies), symbols
    )

    if ai_controller is not None:
        logger.info(
            "AIController enabled: model=%s, interval_loops=%d",
            ai_controller.model,
            ai_controller.interval_loops,
        )
    else:
        logger.info("AIController disabled.")

    if symbol_selector is not None:
        logger.info(
            "SymbolSelector will run every %d loops.",
            symbol_selector.interval_loops,
        )
    else:
        logger.info("SymbolSelector disabled or not configured.")

    # 6) Engine
    engine = Engine(
        client=client,
        strategies=strategies,
        symbols=symbols,
        order_manager=order_manager,
        quote_provider=quote_provider,
        poll_interval=poll_interval,
        paper_cash=paper_cash,
        ai_controller=ai_controller,
        intraday_stats=intraday_stats,
        symbol_selector=symbol_selector,
    )

    logger.info(
        "Engine created (poll_interval=%.2fs, paper_cash=%s)",
        poll_interval,
        "None" if paper_cash is None else f"{paper_cash:.2f}",
    )

    print(
        f"Starting HFTA engine in DRY-RUN mode on account name='HFTA' "
        f"using config: {cfg_path}"
    )
    logger.info(
        "Starting engine loop in DRY-RUN mode on account name='HFTA'."
    )

    # Log any unhandled runtime error from the engine to the log file.
    try:
        engine.run_forever()
    except Exception:
        logger.exception("Unhandled exception in engine.run_forever")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
