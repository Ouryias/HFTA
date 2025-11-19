from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from HFTA.core.execution_tracker import ExecutionTracker
from HFTA.market.intraday_stats import IntradayStatsTracker
from HFTA.strategies.base import Strategy
from HFTA.broker.client import Quote  # for type annotations

# Optional: if you have a MarketUniverse provider (Polygon.io), it can be passed in.
try:
    from HFTA.market.universe import MarketUniverse  # type: ignore
except Exception:
    MarketUniverse = None

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # optional dependency

logger = logging.getLogger(__name__)

@dataclass
class SymbolScore:
    symbol: str
    trade_count: int
    realized_pnl: float
    avg_pnl_per_trade: float
    liquidity_score: float
    day_change_pct: float
    intraday_return: float
    intraday_volatility: float
    intraday_range_pct: float

class SymbolSelector:
    """
    Symbol picker that uses:
      - trading experience (ExecutionTracker per-strategy/per-symbol PnL)
      - market-level metrics (if a MarketUniverse is provided)
      - intraday stats derived from quotes the engine sees

    Modes:
      - 'heuristic': heuristic-only (PnL, liquidity, intraday stats)
      - 'gpt': GPT-only (falls back to heuristic on failure)
      - 'hybrid': GPT first, heuristic as fallback/complement
    """
    def __init__(
        self,
        market_universe: Optional[Any] = None,  # MarketUniverse or None
        interval_loops: int = 60,
        min_trades: int = 3,
        enabled: bool = True,
        mode: str = "hybrid",
        model: str = "gpt-5-mini",
    ) -> None:
        self.market_universe = market_universe
        self.interval_loops = max(1, int(interval_loops))
        self.min_trades = max(1, int(min_trades))
        self.enabled = enabled
        self.mode = mode.lower().strip()
        self.model = model

        self._loop_counter = 0
        self.client: Optional[Any] = None
        self._gpt_enabled: bool = False
        self._init_client_if_needed()

        logger.info(
            "SymbolSelector initialized (enabled=%s, interval_loops=%d, "
            "min_trades=%d, mode=%s, gpt_enabled=%s)",
            self.enabled, self.interval_loops, self.min_trades,
            self.mode, self._gpt_enabled,
        )

    def _init_client_if_needed(self) -> None:
        if self.mode not in {"gpt", "hybrid"}:
            self.client = None
            self._gpt_enabled = False
            return
        if OpenAI is None:
            logger.warning("SymbolSelector: openai package not installed; falling back to heuristic.")
            self.client = None
            self._gpt_enabled = False
            return
        api_key = (
            os.getenv("HFTA_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("openai_api_key")
        )
        if not api_key:
            logger.warning("SymbolSelector: no OpenAI API key found; falling back to heuristic mode.")
            self.client = None
            self._gpt_enabled = False
            return
        try:
            self.client = OpenAI(api_key=api_key)
            self._gpt_enabled = True
        except Exception as exc:
            logger.warning("SymbolSelector: failed to initialize OpenAI client: %s; falling back to heuristic mode.", exc, exc_info=True)
            self.client = None
            self._gpt_enabled = False

    # ------------------------------------------------------------------ #
    # Engine entry point
    # ------------------------------------------------------------------ #
    def on_loop(
        self,
        *,
        strategies: List[Strategy],
        tracker: Optional[ExecutionTracker],
        intraday_stats: Optional[IntradayStatsTracker] = None,
    ) -> None:
        """Run symbol selection logic every `interval_loops` engine iterations."""
        if not self.enabled:
            return
        self._loop_counter += 1
        if self._loop_counter % self.interval_loops != 0:
            return
        if tracker is None:
            logger.debug("SymbolSelector: no ExecutionTracker; skipping.")
            return

        # Collect performance stats per strategy/symbol
        per_strat_stats = tracker.per_strategy_symbol_summary()
        if not per_strat_stats:
            logger.debug("SymbolSelector: no per-strategy stats yet; skipping.")
            return

        # Determine full universe and candidates (non-active symbols)
        if self.market_universe is not None and getattr(self.market_universe, "symbols", None):
            symbol_universe: Set[str] = {s.upper() for s in self.market_universe.symbols}
            market_metrics: Mapping[str, Mapping[str, float]] = getattr(self.market_universe, "metrics_by_symbol", {})
        else:
            symbol_universe = {s.upper() for s in getattr(self, "universe_symbols", [])} if hasattr(self, "universe_symbols") else self._collect_fallback_universe(per_strat_stats, strategies)
            market_metrics = {}

        active_symbols: Set[str] = set()
        for strat in strategies:
            sym = getattr(strat, "symbol", None)
            if sym is None:
                continue
            if isinstance(sym, (list, tuple)):
                for s in sym:
                    active_symbols.add(str(s).upper())
            else:
                active_symbols.add(str(sym).upper())
        candidates: Set[str] = symbol_universe - active_symbols

        if not symbol_universe:
            logger.debug("SymbolSelector: empty symbol universe; skipping.")
            return
        if not candidates:
            logger.debug("SymbolSelector: no candidate symbols to evaluate; skipping.")
            return

        # Fetch intraday metrics and latest prices for candidate symbols
        intraday_metrics: Dict[str, Any] = {}
        if intraday_stats is not None:
            try:
                intraday_metrics = intraday_stats.summary() or {}
            except Exception as exc:
                logger.warning("SymbolSelector: intraday_stats summary unavailable: %s", exc)
                intraday_metrics = {}
        quotes_by_symbol: Dict[str, Quote] = {}
        if self.client is None:  # if no external API, use quote_provider via engine
            # Assuming the Engine passes a QuoteProvider attached to this instance if needed
            try:
                # If a QuoteProvider integration existed, we would call it here
                pass  # (Engine handles quote fetching now; candidates only fetched if external data needed)
            except Exception as exc:
                logger.warning("SymbolSelector: candidate quote fetch failed: %s", exc, exc_info=True)
        # If symbol_selector were provided a quote_provider, one could integrate like:
        # quotes_by_symbol = self.quote_provider.get_quotes(list(candidates))

        # For demonstration, if engine already fetched intraday stats for active symbols,
        # we might manually fetch one quote per candidate (omitted here).
        # Instead, rely on any market_metrics (e.g., last close for day_change) and assume minimal momentum if unavailable.

        # Update metrics using any available quotes (e.g., via Universe or prior runs)
        if not hasattr(self, "prev_prices"):
            self.prev_prices: Dict[str, float] = {}
        if quotes_by_symbol:
            for sym_u, quote in quotes_by_symbol.items():
                price: Optional[float] = quote.last
                if price is None:
                    if quote.bid is not None and quote.ask is not None:
                        price = (quote.bid + quote.ask) / 2.0
                    elif quote.bid is not None:
                        price = quote.bid
                    elif quote.ask is not None:
                        price = quote.ask
                if price is None or price <= 0.0:
                    continue
                # Update day_change% if last close known
                if sym_u in market_metrics:
                    try:
                        last_close_val = float(market_metrics[sym_u].get("close") or market_metrics[sym_u].get("price") or 0.0)
                    except Exception:
                        last_close_val = None
                    if last_close_val and last_close_val > 0.0:
                        market_metrics[sym_u]["day_change_pct"] = (price / last_close_val - 1.0) * 100.0
                # Compute short-term momentum since last selection
                momentum_pct: Optional[float] = None
                if sym_u in self.prev_prices and self.prev_prices[sym_u] > 0.0:
                    momentum_pct = (price / self.prev_prices[sym_u] - 1.0) * 100.0
                self.prev_prices[sym_u] = price
                if momentum_pct is not None:
                    intraday_metrics[sym_u] = {
                        "intraday_return": momentum_pct,
                        "volatility": 0.0,
                        "range_pct": 0.0
                    }

        # Score all symbols (active + candidates) and decide assignments
        scores: Dict[str, SymbolScore] = self._compute_symbol_scores(
            per_strat_stats=per_strat_stats,
            symbol_universe=symbol_universe,
            market_metrics=market_metrics,
            intraday_metrics=intraday_metrics,
        )
        if not scores:
            logger.debug("SymbolSelector: no symbol scores computed; skipping.")
            return

        decisions: Dict[str, str] = {}
        use_gpt = self._gpt_enabled and self.mode in {"gpt", "hybrid"}
        if use_gpt:
            try:
                decisions = self._pick_via_gpt(
                    per_strat_stats=per_strat_stats,
                    symbol_universe=symbol_universe,
                    strategies=strategies,
                    market_metrics=market_metrics,
                    intraday_metrics=intraday_metrics,
                )
            except Exception:
                logger.exception("SymbolSelector: GPT-based selection failed")

        if not decisions or self.mode in {"heuristic", "hybrid"}:
            def _total_score(sym_score: SymbolScore) -> float:
                return (
                    3.0 * sym_score.realized_pnl
                    + 2.0 * sym_score.avg_pnl_per_trade
                    + 1.0 * sym_score.liquidity_score
                    + 0.5 * sym_score.day_change_pct
                    + 1.5 * sym_score.intraday_range_pct
                    + 1.0 * sym_score.intraday_volatility
                    + 1.0 * sym_score.intraday_return
                )
            best_symbol = max(scores.values(), key=_total_score).symbol
            for strat in strategies:
                decisions.setdefault(strat.name, best_symbol)

        if not decisions:
            logger.info("SymbolSelector: no symbol decisions made this round.")
            return

        # Log decisions and reassign symbols
        for strat in strategies:
            target = decisions.get(strat.name)
            if not target:
                continue
            current = getattr(strat, "symbol", None)
            if isinstance(target, (list, tuple)):
                target_syms = [str(s).upper() for s in target]
                target_display = ", ".join(target_syms)
            else:
                target_syms = None
                target_display = str(target).upper()
            if current is None:
                changed = True
            elif isinstance(current, (list, tuple)):
                current_set = {str(s).upper() for s in current}
                target_set = set(target_syms) if target_syms is not None else {target_display}
                changed = (current_set != target_set)
            else:
                changed = (str(current).upper() != target_display)
            metric_info = ""
            if target_syms is None:
                sc = scores.get(target_display)
                if sc is not None:
                    metric_info = (
                        f"(trade_count={sc.trade_count}, realized_pnl={sc.realized_pnl:.2f}, avg_pnl={sc.avg_pnl_per_trade:.2f}, "
                        f"liquidity={sc.liquidity_score:.2f}, day_change={sc.day_change_pct:.2f}%, intraday_return={sc.intraday_return:.2f}%, "
                        f"volatility={sc.intraday_volatility:.2f}, range={sc.intraday_range_pct:.2f}%)"
                    )
            if changed:
                logger.info("SymbolSelector: strategy '%s' reassigning from %s to %s %s",
                            strat.name, current, target_display, metric_info)
            else:
                logger.info("SymbolSelector: strategy '%s' remains on %s %s",
                            strat.name, target_display, metric_info)
            if changed:
                setattr(strat, "symbol", target_syms if target_syms is not None else target_display)

        # Persist state to JSON file
        record: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "active_symbols": sorted(active_symbols),
            "candidate_symbols": sorted(candidates),
            "decisions": decisions,
            "scores": {}
        }
        for sym_u, sc in scores.items():
            record["scores"][sym_u] = {
                "trade_count": sc.trade_count,
                "realized_pnl": sc.realized_pnl,
                "avg_pnl_per_trade": sc.avg_pnl_per_trade,
                "liquidity_score": sc.liquidity_score,
                "day_change_pct": sc.day_change_pct,
                "intraday_return": sc.intraday_return,
                "intraday_volatility": sc.intraday_volatility,
                "intraday_range_pct": sc.intraday_range_pct,
            }
            record["scores"][sym_u]["total_score"] = (
                3.0 * sc.realized_pnl
                + 2.0 * sc.avg_pnl_per_trade
                + 1.0 * sc.liquidity_score
                + 0.5 * sc.day_change_pct
                + 1.5 * sc.intraday_range_pct
                + 1.0 * sc.intraday_volatility
                + 1.0 * sc.intraday_return
            )
        state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "logs", "symbol_selection_state.json")
        try:
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
        except Exception:
            pass
        try:
            with open(state_file, "a") as f:
                json.dump(record, f)
                f.write("\n")
        except Exception as exc:
            logger.warning("SymbolSelector: failed to write state file: %s", exc)

    # ------------------------------------------------------------------ #
    # Universe helpers
    # ------------------------------------------------------------------ #
    def _collect_fallback_universe(
        self,
        per_strat_stats: Mapping[str, Mapping[str, Mapping[str, float]]],
        strategies: List[Strategy],
    ) -> Set[str]:
        symbols: Set[str] = set()
        for sym_map in per_strat_stats.values():
            for sym in sym_map.keys():
                symbols.add(sym.upper())
        for strat in strategies:
            sym = getattr(strat, "symbol", None)
            if sym:
                symbols.add(strat.symbol.upper())
        return symbols

    # ------------------------------------------------------------------ #
    # Heuristic selection (detailed in on_loop)
    # ------------------------------------------------------------------ #
    def _compute_symbol_scores(
        self,
        per_strat_stats: Mapping[str, Mapping[str, Mapping[str, float]]],
        symbol_universe: Set[str],
        market_metrics: Mapping[str, Mapping[str, float]],
        intraday_metrics: Mapping[str, Mapping[str, float]],
    ) -> Dict[str, SymbolScore]:
        """
        Blend:
          - experience (PnL, trade_count)
          - baseline liquidity (if available)
          - intraday volatility / range / return
        """
        # Aggregate experience across strategies
        agg_tc: Dict[str, int] = {}
        agg_pnl: Dict[str, float] = {}
        for sym_map in per_strat_stats.values():
            for symbol, stats in sym_map.items():
                symbol_u = symbol.upper()
                if symbol_u not in symbol_universe:
                    continue
                tc = int(stats.get("trade_count", 0))
                pnl = float(stats.get("realized_pnl", 0.0))
                if tc <= 0:
                    continue
                agg_tc[symbol_u] = agg_tc.get(symbol_u, 0) + tc
                agg_pnl[symbol_u] = agg_pnl.get(symbol_u, 0.0) + pnl

        scores: Dict[str, SymbolScore] = {}
        for symbol_u in symbol_universe:
            tc = agg_tc.get(symbol_u, 0)
            pnl = agg_pnl.get(symbol_u, 0.0)
            if tc < self.min_trades:
                pnl = 0.0
            avg = pnl / tc if tc > 0 else 0.0

            m = market_metrics.get(symbol_u, {})
            dollar_vol = float(m.get("dollar_volume", 0.0))
            day_chg = float(m.get("day_change_pct", 0.0))
            liq = math.log10(dollar_vol + 1.0) if dollar_vol > 0.0 else 0.0

            i = intraday_metrics.get(symbol_u, {})
            i_ret = float(i.get("intraday_return", 0.0))
            i_vol = float(i.get("volatility", 0.0))
            i_range = float(i.get("range_pct", 0.0))

            scores[symbol_u] = SymbolScore(
                symbol=symbol_u,
                trade_count=tc,
                realized_pnl=pnl,
                avg_pnl_per_trade=avg,
                liquidity_score=liq,
                day_change_pct=day_chg,
                intraday_return=i_ret,
                intraday_volatility=i_vol,
                intraday_range_pct=i_range,
            )
        return scores

    def _pick_via_gpt(
        self,
        per_strat_stats: Mapping[str, Mapping[str, Mapping[str, float]]],
        symbol_universe: Set[str],
        strategies: List[Strategy],
        market_metrics: Mapping[str, Mapping[str, float]],
        intraday_metrics: Mapping[str, Mapping[str, float]],
    ) -> Dict[str, str]:
        # Build state JSON for GPT prompt
        state = {
            "symbol_universe": sorted(list(symbol_universe)),
            "strategies": [
                {
                    "name": strat.name,
                    "current_symbol": getattr(strat, "symbol", None),
                    "config": getattr(strat, "config", {}),
                }
                for strat in strategies
            ],
            "per_strategy_symbol_stats": per_strat_stats,
            "market_metrics": market_metrics,
            "intraday_metrics": intraday_metrics,
        }
        state_json = json.dumps(state, separators=(",", ":"), sort_keys=True)
        logger.debug("SymbolSelector: sending state to GPT: %s", state_json)
        if self.client is None:
            raise RuntimeError("SymbolSelector client not initialized")

        # Construct prompts for GPT
        system_prompt = (
            "You are an expert intraday symbol allocator for a trading system. "
            "Decide the single best equity symbol for each strategy based on performance and market conditions."
        )
        user_prompt = (
            "JSON input describes:\n"
            "- symbol_universe: list of allowed tickers\n"
            "- strategies: list of {name, current_symbol, config}\n"
            "- per_strategy_symbol_stats: realized PnL and trade counts\n"
            "- market_metrics: price, dollar_volume, day_change_pct per symbol\n"
            "- intraday_metrics: intraday_return, range_pct, volatility per symbol\n\n"
            "Goals:\n"
            "1) For each strategy, pick the symbol to maximize expected profit.\n"
            "2) Favor symbols with strong PnL, high liquidity, healthy intraday volatility.\n"
            "3) Avoid switching unless new choice is clearly better.\n"
            "4) Only use symbols from symbol_universe.\n\n"
            "Return ONLY a JSON object with format:\n"
            "{ \"decisions\": [ {\"strategy_name\": \"...\", \"target_symbol\": \"...\"}, ... ] }\n"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": f"Current state JSON:\n{state_json}"},
        ]
        try:
            resp = self.client.chat.completions.create(model=self.model, messages=messages)
        except Exception as exc:
            logger.warning("SymbolSelector: OpenAI request failed: %s", exc, exc_info=True)
            return {}

        msg = resp.choices[0].message
        content_str = (msg.content or "").strip() if hasattr(msg, "content") else str(msg).strip()
        if not content_str:
            logger.warning("SymbolSelector: empty response from model")
            return {}
        # Attempt to parse JSON from response
        try:
            parsed = json.loads(content_str)
        except json.JSONDecodeError:
            # Try to extract JSON substring if extra text
            start = content_str.find("{")
            end = content_str.rfind("}")
            if start == -1 or end == -1:
                logger.warning("SymbolSelector: model returned non-JSON output: %r", content_str)
                return {}
            try:
                parsed = json.loads(content_str[start:end+1])
            except Exception as e:
                logger.warning("SymbolSelector: could not parse JSON from model output: %s", e)
                return {}
        if not isinstance(parsed, Mapping):
            return {}
        raw_decisions = parsed.get("decisions", [])
        if not isinstance(raw_decisions, list):
            logger.warning("SymbolSelector: 'decisions' field not a list; got %r", raw_decisions)
            return {}

        decisions: Dict[str, str] = {}
        allowed_set = {s.upper() for s in symbol_universe}
        for item in raw_decisions:
            if not isinstance(item, Mapping):
                continue
            sname = item.get("strategy_name") or item.get("strategy")
            sym = item.get("target_symbol") or item.get("symbol")
            if not sname or not sym:
                continue
            sym_u = str(sym).upper()
            if sym_u not in allowed_set:
                logger.debug("SymbolSelector: ignoring GPT suggestion %s -> %s (not in universe)", sname, sym_u)
                continue
            decisions[str(sname)] = sym_u
        return decisions
