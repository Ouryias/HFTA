# scripts/run_backtest.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from HFTA.core.risk_manager import RiskConfig
from HFTA.sim.backtester import BacktestConfig, BacktestEngine
from HFTA.strategies.base import Strategy
from HFTA.strategies.micro_market_maker import MicroMarketMaker
from HFTA.strategies.micro_trend_scalper import MicroTrendScalper


STRATEGY_REGISTRY = {
    "micro_market_maker": MicroMarketMaker,
    "micro_trend_scalper": MicroTrendScalper,
}


def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r") as f:
        return json.load(f)


def build_risk_config(cfg: dict) -> RiskConfig:
    risk = cfg.get("risk", {})
    return RiskConfig(
        max_notional_per_order=risk.get("max_notional_per_order", 100.0),
        max_cash_utilization=risk.get("max_cash_utilization", 0.1),
        allow_short_selling=risk.get("allow_short_selling", False),
    )


def build_strategies(cfg: dict) -> List[Strategy]:
    """
    Expected config shape example:

    "strategies": [
      {
        "name": "mm_AAPL",
        "type": "micro_market_maker",
        "config": { ... }
      },
      {
        "name": "ts_AAPL",
        "type": "micro_trend_scalper",
        "config": { ... }
      }
    ]
    """
    out: List[Strategy] = []
    for s in cfg.get("strategies", []):
        name = s["name"]
        type_key = s["type"]
        strat_cls = STRATEGY_REGISTRY[type_key]
        strat_cfg = s.get("config", {})
        out.append(strat_cls(name=name, config=strat_cfg))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline HFTA backtest.")
    parser.add_argument(
        "--config",
        default="configs/paper_aapl.json",
        help="Path to JSON config (default: configs/paper_aapl.json)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Number of simulated timesteps (default: 2000)",
    )
    args = parser.parse_args()

    cfg_json = load_json(args.config)
    symbols = cfg_json.get("symbols") or ["AAPL"]
    symbol = symbols[0].upper()

    paper_cash = float(cfg_json.get("paper_cash", 100_000.0))
    poll_interval = int(cfg_json.get("poll_interval", 5))

    risk_cfg = build_risk_config(cfg_json)
    strategies = build_strategies(cfg_json)

    bt_cfg = BacktestConfig(
        symbol=symbol,
        starting_price=float(cfg_json.get("starting_price", 40.0)),
        starting_cash=paper_cash,
        steps=args.steps,
        step_seconds=poll_interval,
        volatility_annual=float(cfg_json.get("volatility_annual", 0.4)),
        spread_cents=float(cfg_json.get("spread_cents", 0.10)),
        risk_config=risk_cfg,
    )

    engine = BacktestEngine(strategies=strategies, config=bt_cfg)
    result = engine.run()

    print("=== BACKTEST SUMMARY ===")
    print(f"Symbol: {result.symbol}")
    print(f"Starting cash: {result.starting_cash:,.2f}")
    print(f"Final cash: {result.final_cash:,.2f}")
    print(f"Final equity: {result.final_equity:,.2f}")
    print(f"Realized PnL: {result.realized_pnl:,.2f}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Steps simulated: {len(result.equity_curve)}")
    print("Open positions at end:")
    for sym, pos in result.positions_summary.items():
        print(
            f"  {sym}: qty={pos.quantity:.2f}, "
            f"avg_price={pos.avg_price:.2f}, "
            f"realized_pnl={pos.realized_pnl:.2f}"
        )


if __name__ == "__main__":
    main()
