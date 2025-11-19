# HFTA/config_loader.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from HFTA.core.risk_manager import RiskConfig
from HFTA.strategies.base import Strategy
from HFTA.strategies.micro_market_maker import MicroMarketMaker
from HFTA.strategies.micro_trend_scalper import MicroTrendScalper


# Registry of strategy types → classes.
STRATEGY_REGISTRY = {
    "micro_market_maker": MicroMarketMaker,
    "micro_trend_scalper": MicroTrendScalper,
}


@dataclass
class LoadedConfig:
    """
    Normalized config shared by engine and backtester.
    """
    raw: Dict[str, Any]

    symbols: List[str]
    paper_cash: float
    poll_interval: float

    risk_config: RiskConfig
    strategies: List[Strategy]

    # Optional / engine-only blocks (may be empty dicts)
    ai_block: Dict[str, Any]
    data_block: Dict[str, Any]
    universe_block: Dict[str, Any]
    symbol_selector_block: Dict[str, Any]


def _build_risk_config(raw_cfg: Dict[str, Any]) -> RiskConfig:
    risk_dict = raw_cfg.get("risk", {}) or {}

    # RiskConfig has defaults; **risk_dict lets you add new fields like
    # max_total_exposure_ratio and max_positions without breaking older configs.
    return RiskConfig(**risk_dict)


def _build_strategies(raw_cfg: Dict[str, Any]) -> List[Strategy]:
    strategies_cfg = raw_cfg.get("strategies", []) or []
    strategies: List[Strategy] = []

    for entry in strategies_cfg:
        s_type = entry.get("type")
        s_name = entry.get("name")
        s_conf = entry.get("config", {}) or {}

        if not s_type or not s_name:
            raise ValueError(f"Invalid strategy entry (missing type/name): {entry}")

        cls = STRATEGY_REGISTRY.get(s_type)
        if cls is None:
            raise ValueError(f"Unknown strategy type {s_type!r} in config")

        strat = cls(name=s_name, config=s_conf)
        strategies.append(strat)

    return strategies


def load_config(path: str | Path) -> LoadedConfig:
    """
    Load a JSON config file and construct RiskConfig + Strategy instances.

    The JSON is expected to have at least:

        {
          "symbols": [...],
          "paper_cash": 100000,
          "poll_interval": 5,
          "risk": { ... },
          "strategies": [
              {"type": "...", "name": "...", "config": {...}},
              ...
          ],

          "data": {...},             # optional, engine only
          "universe": {...},         # optional, engine only
          "ai": {...},               # optional, engine only
          "symbol_selector": {...}   # optional, engine only
        }
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    symbols = [str(s).upper() for s in raw.get("symbols", [])]
    if not symbols:
        raise ValueError("Config must define a non-empty 'symbols' list")

    paper_cash = float(raw.get("paper_cash", 0.0))
    poll_interval = float(raw.get("poll_interval", 1.0))

    risk_config = _build_risk_config(raw)
    strategies = _build_strategies(raw)

    ai_block = raw.get("ai", {}) or {}
    data_block = raw.get("data", {}) or {}
    universe_block = raw.get("universe", {}) or {}
    symbol_selector_block = raw.get("symbol_selector", {}) or {}

    return LoadedConfig(
        raw=raw,
        symbols=symbols,
        paper_cash=paper_cash,
        poll_interval=poll_interval,
        risk_config=risk_config,
        strategies=strategies,
        ai_block=ai_block,
        data_block=data_block,
        universe_block=universe_block,
        symbol_selector_block=symbol_selector_block,
    )
