# scripts/run_backtest.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from HFTA.logging_utils import setup_logging, parse_log_level
from HFTA.sim.backtester import BacktestConfig, BacktestEngine, load_quotes_from_csv, generate_random_walk_quotes
from HFTA.config_loader import load_config

logger = logging.getLogger(__name__)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run HFTA backtest.")
    parser.add_argument(
        "--config",
        default="configs/paper_aapl.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of synthetic steps when not using --quotes-csv.",
    )
    parser.add_argument(
        "--quotes-csv",
        default=None,
        help="Optional CSV of historical quotes to backtest on.",
    )
    parser.add_argument(
        "--equity-csv",
        default=None,
        help="Optional path to write equity curve CSV.",
    )
    parser.add_argument(
        "--fills-csv",
        default=None,
        help="Optional path to write fills CSV.",
    )
    parser.add_argument(
        "--log-file",
        default="logs/backtest.log",
        help="Path to log file.",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        help="Log level (DEBUG, INFO, WARNING, ERROR).",
    )
    args = parser.parse_args(argv)

    log_level = parse_log_level(args.log_level)
    setup_logging("HFTA.backtest", args.log_file, level=log_level)
    logging.getLogger("peewee").setLevel(logging.WARNING)

    logger.info("Starting backtest with config=%s", args.config)

    loaded = load_config(args.config)

    # Build BacktestConfig
    backtest_cfg = BacktestConfig(
        symbol=loaded.symbols[0],
        starting_cash=loaded.paper_cash,
        risk_config=loaded.risk_config,
        steps=args.steps,
        step_seconds=loaded.poll_interval,
    )

    # Quotes
    if args.quotes_csv:
        quotes = load_quotes_from_csv(Path(args.quotes_csv))
        logger.info("Loaded %d quotes from %s", len(quotes), args.quotes_csv)
    else:
        quotes = generate_random_walk_quotes(
            symbol=backtest_cfg.symbol,
            starting_price=loaded.raw.get("starting_price", 150.0),
            steps=args.steps,
            step_seconds=backtest_cfg.step_seconds,
            volatility_annual=loaded.raw.get("volatility_annual", 0.2),
            spread_cents=loaded.raw.get("spread_cents", 1.0),
        )
        logger.info(
            "Generated %d synthetic quotes for symbol=%s",
            len(quotes),
            backtest_cfg.symbol,
        )

    engine = BacktestEngine(
        strategies=loaded.strategies,
        config=backtest_cfg,
        quotes=quotes,
    )

    result = engine.run()

    logger.info(
        "BACKTEST SUMMARY: starting_cash=%.2f final_equity=%.2f realized_pnl=%.2f max_drawdown=%.2f",
        result.starting_cash,
        result.final_equity,
        result.realized_pnl,
        result.max_drawdown,
    )

    if args.equity_csv:
        result.write_equity_csv(Path(args.equity_csv))
        logger.info("Wrote equity curve to %s", args.equity_csv)

    if args.fills_csv:
        result.write_fills_csv(Path(args.fills_csv))
        logger.info("Wrote fills blotter to %s", args.fills_csv)


if __name__ == "__main__":
    main()
