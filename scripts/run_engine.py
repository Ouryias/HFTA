# scripts/run_engine.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from HFTA.logging_utils import setup_logging, parse_log_level
from HFTA.broker.client import WealthsimpleClient
from HFTA.core.risk_manager import RiskConfig, RiskManager
from HFTA.core.order_manager import OrderManager
from HFTA.core.execution_tracker import ExecutionTracker
from HFTA.core.engine import Engine
from HFTA.ai.controller import AIController
from HFTA.strategies.micro_market_maker import MicroMarketMaker
from HFTA.strategies.micro_trend_scalper import MicroTrendScalper


STRATEGY_REGISTRY = {
    "micro_market_maker": MicroMarketMaker,
    "micro_trend_scalper": MicroTrendScalper,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HFTA trading engine (DRY-RUN).")
    parser.add_argument(
        "--config",
        default="configs/paper_aapl.json",
        help="Path to JSON config file (default: configs/paper_aapl.json)",
    )
    parser.add_argument(
        "--log-file",
        default="logs/engine.log",
        help="Path to log file (default: logs/engine.log)",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: DEBUG",
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_risk_config(cfg: Dict[str, Any]) -> RiskConfig:
    risk_cfg = cfg.get("risk", {})
    return RiskConfig(
        max_notional_per_order=float(risk_cfg.get("max_notional_per_order", 1000.0)),
        max_cash_utilization=float(risk_cfg.get("max_cash_utilization", 0.1)),
        allow_short_selling=bool(risk_cfg.get("allow_short_selling", False)),
    )


def build_strategies(cfg: Dict[str, Any]) -> List[Any]:
    """
    Build strategy instances from config.

    Primary path (matches original repo):
        StrategyClass(name=<name>, config=<dict>)

    Fallback (for any future strategy classes that take explicit kwargs):
        StrategyClass(name=<name>, **config_dict)
    """
    strategies_cfg = cfg.get("strategies", [])
    strategies: List[Any] = []

    for strat_cfg in strategies_cfg:
        s_type = strat_cfg["type"]
        s_name = strat_cfg["name"]
        s_conf = strat_cfg.get("config", {})

        cls = STRATEGY_REGISTRY.get(s_type)
        if cls is None:
            raise ValueError(f"Unknown strategy type: {s_type}")

        try:
            strat = cls(name=s_name, config=s_conf)
        except TypeError:
            strat = cls(name=s_name, **s_conf)

        strategies.append(strat)

    return strategies


def main() -> None:
    args = parse_args()
    level = parse_log_level(args.log_level)

    logger = setup_logging(
        "HFTA.engine",
        log_file=args.log_file,
        level=level,
        log_to_console=True,
    )

    logger.debug("Parsed arguments: %s", vars(args))

    try:
        # ------------------------------------------------------------------
        # Load config
        # ------------------------------------------------------------------
        cfg = load_config(args.config)
        logger.info("Starting run engine with config=%s", args.config)
        logger.debug("Loaded config from %s", args.config)

        symbols = cfg.get("symbols", [])
        poll_interval = float(cfg.get("poll_interval", 5.0))
        paper_cash = float(cfg.get("paper_cash", 100000.0))
        account_name = cfg.get("account_name", "HFTA")

        risk_config = build_risk_config(cfg)
        logger.info("RiskConfig: %s", risk_config)

        # ------------------------------------------------------------------
        # Broker client (Wealthsimple)
        # ------------------------------------------------------------------
        # WealthsimpleClient internally handles authentication and default
        # account selection; we just instantiate it.
        client = WealthsimpleClient()
        logger.info(
            "WealthsimpleClient created (will use its configured default account, "
            "expected name=%r)",
            account_name,
        )

        # ------------------------------------------------------------------
        # Core components
        # ------------------------------------------------------------------
        execution_tracker = ExecutionTracker()
        risk_manager = RiskManager(risk_config)
        order_manager = OrderManager(
            client=client,
            risk_manager=risk_manager,
            execution_tracker=execution_tracker,
            live=False,  # DRY-RUN for now
        )

        # ------------------------------------------------------------------
        # Strategies
        # ------------------------------------------------------------------
        strategies = build_strategies(cfg)
        logger.info("Built %d strategies for symbols=%s", len(strategies), symbols)

        # ------------------------------------------------------------------
        # AI controller (optional)
        # ------------------------------------------------------------------
        ai_cfg = cfg.get("ai", {})
        ai_enabled = bool(ai_cfg.get("enabled", False))
        ai_model = ai_cfg.get("model", "gpt-5-mini")
        ai_interval = int(ai_cfg.get("interval_loops", 12))
        ai_temperature = float(ai_cfg.get("temperature", 0.2))
        ai_max_tokens = int(ai_cfg.get("max_output_tokens", 512))

        ai_controller = AIController(
            model=ai_model,
            interval_loops=ai_interval,
            temperature=ai_temperature,
            max_output_tokens=ai_max_tokens,
            enabled=ai_enabled,
        )

        # ------------------------------------------------------------------
        # Engine
        # ------------------------------------------------------------------
        engine = Engine(
            client=client,
            strategies=strategies,
            order_manager=order_manager,
            ai_controller=ai_controller,
            poll_interval=poll_interval,
            paper_cash=paper_cash,
            symbols=symbols,
            logger=logger,
        )

        logger.info(
            "Engine created (poll_interval=%.3fs, paper_cash=%.2f, symbols=%s)",
            poll_interval,
            paper_cash,
            symbols,
        )

        print(
            f"Starting HFTA engine in DRY-RUN mode on account name='{account_name}' "
            f"using config: {args.config}"
        )

        # Run engine with exception logging so stack traces go to the log file.
        try:
            engine.run_forever()
        except KeyboardInterrupt:
            print("\nStopped by user.")
            logger.info("Engine stopped by user (KeyboardInterrupt).")
        except Exception:
            logger.exception("Unhandled exception in engine.run_forever")
            raise

    except Exception:
        # Any setup/runtime error in main() will be logged here as ERROR/CRITICAL.
        logger.exception("Unhandled exception in run_engine.main")
        raise


if __name__ == "__main__":
    main()
