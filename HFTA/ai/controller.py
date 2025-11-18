from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


class AIController:
    """
    Periodically calls a GPT model to get small parameter tweaks and
    high-level commentary on how the system is doing.

    Design goals:
    - If the OpenAI SDK is missing or the API misbehaves, the engine must
      keep running. All failures here are logged but not fatal.
    - The model only receives a compact JSON snapshot of:
        * realized PnL
        * open positions
        * a few risk limits
        * a few strategy parameters
    - The model may suggest:
        * strategy_updates: list of {name, params}
        * risk_updates: dict of updated numeric fields
    - Parameter changes are kept small (clamped) and only applied to
      existing numeric attrs. Shorts are never enabled.
    """

    def __init__(
        self,
        model: str,
        interval_loops: int = 12,
        temperature: float = 0.2,
        max_output_tokens: int = 512,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.interval_loops = max(1, int(interval_loops))
        self.temperature = float(temperature)  # currently unused
        self.max_output_tokens = int(max_output_tokens)

        self._loop_counter = 0

        # OpenAI client setup
        if not enabled:
            self.enabled = False
            self.client = None
            logger.info("AIController disabled in config.")
            return

        if OpenAI is None:
            self.enabled = False
            self.client = None
            logger.warning(
                "AIController disabled: openai package is not installed."
            )
            return

        api_key = (
            os.getenv("HFTA_OPENAI_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("openai_api_key")  # just in case
        )

        if not api_key:
            self.enabled = False
            self.client = None
            logger.warning(
                "AIController disabled: no OpenAI API key found. "
                "Set HFTA_OPENAI_API_KEY or OPENAI_API_KEY in your environment "
                "to enable AI tuning."
            )
            return

        # If we reach here, AI is enabled
        self.client = OpenAI(api_key=api_key)
        self.enabled = True
        logger.info(
            "AIController initialized with model=%s, interval_loops=%d",
            self.model,
            self.interval_loops,
        )

    # ------------------------------------------------------------------ #
    # Entry point from Engine
    # ------------------------------------------------------------------ #

    def on_loop(
        self,
        *,
        strategies: List[Any],
        risk_config: Any,
        tracker: Any,
    ) -> None:
        """
        Called by Engine once per event-loop iteration via keyword args.
        """
        self.maybe_run(
            risk_config=risk_config,
            strategies=strategies,
            execution_tracker=tracker,
        )

    # ------------------------------------------------------------------ #
    # Periodic driver
    # ------------------------------------------------------------------ #

    def maybe_run(
        self,
        risk_config: Any,
        strategies: List[Any],
        execution_tracker: Any,
    ) -> None:
        if not self.enabled or self.client is None:
            return

        self._loop_counter += 1
        if self._loop_counter % self.interval_loops != 0:
            return

        try:
            state_json = self._build_state_json(
                risk_config=risk_config,
                strategies=strategies,
                execution_tracker=execution_tracker,
            )
            logger.debug("AIController state JSON: %s", state_json)

            ai_raw = self._call_model(state_json)
            logger.debug("AIController raw parsed response: %s", ai_raw)

            self._apply_response(
                response=ai_raw,
                risk_config=risk_config,
                strategies=strategies,
            )
        except Exception as exc:
            # Never crash the engine because of AI issues
            logger.warning("AIController error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------ #
    # Build state snapshot
    # ------------------------------------------------------------------ #

    def _build_state_json(
        self,
        risk_config: Any,
        strategies: List[Any],
        execution_tracker: Any,
    ) -> str:
        """
        Build a compact JSON string of:
            - total realized PnL
            - per-symbol positions
            - selected risk_config fields
            - selected strategy parameters
        """
        state: Dict[str, Any] = {}

        # Positions / realized PnL
        realized_total = 0.0
        positions_out: Dict[str, Any] = {}

        try:
            positions = getattr(execution_tracker, "positions", {}) or {}
            realized_map = getattr(
                execution_tracker, "realized_pnl_per_symbol", {}
            ) or {}

            for symbol, pos in positions.items():
                qty = float(getattr(pos, "quantity", 0.0))
                avg_price = float(getattr(pos, "avg_price", 0.0))
                realized = float(realized_map.get(symbol, 0.0))
                realized_total += realized

                positions_out[symbol] = {
                    "quantity": qty,
                    "avg_price": avg_price,
                    "realized_pnl": realized,
                }
        except Exception as exc:
            logger.debug(
                "AIController: failed to extract positions from tracker: %s",
                exc,
                exc_info=True,
            )

        state["realized_pnl_total"] = realized_total
        state["positions"] = positions_out

        # Risk config (only simple fields)
        risk_info: Dict[str, Any] = {}
        for key in (
            "max_notional_per_order",
            "max_cash_utilization",
            "allow_short_selling",
        ):
            if hasattr(risk_config, key):
                val = getattr(risk_config, key)
                if isinstance(val, (int, float, bool)):
                    risk_info[key] = val
        state["risk"] = risk_info

        # Strategies snapshot
        strat_snap: List[Dict[str, Any]] = []
        numeric_fields = [
            "spread",
            "min_spread",
            "max_spread",
            "vol_window",
            "vol_to_spread",
            "order_quantity",
            "max_inventory",
            "short_window",
            "long_window",
            "trend_threshold",
            "trailing_stop_pct",
            "take_profit_pct",
            "max_position",
        ]

        for strat in strategies:
            info: Dict[str, Any] = {}
            info["name"] = getattr(strat, "name", None)
            info["type"] = strat.__class__.__name__

            for field in numeric_fields:
                if hasattr(strat, field):
                    val = getattr(strat, field)
                    if isinstance(val, (int, float)):
                        info[field] = float(val)

            strat_snap.append(info)

        state["strategies"] = strat_snap

        return json.dumps(state, sort_keys=True)

    # ------------------------------------------------------------------ #
    # Call OpenAI
    # ------------------------------------------------------------------ #

    def _call_model(self, state_json: str) -> Mapping[str, Any]:
        """
        Call the model and return a parsed JSON object.

        Notes:
        - We DO NOT pass temperature, because this has produced
          `unsupported_value` errors for some chat models.
        - We DO NOT use `response_format` / JSON mode, since content shapes
          have been changing. We instead ask for JSON in plain text and parse.
        - If anything goes wrong, we log and return an empty dict.
        """
        if self.client is None:
            raise RuntimeError("AIController client not initialized")

        system_prompt = (
            "You are a cautious but proactive trading-parameter assistant for a "
            "small intraday equities system running on paper only. "
            "You must preserve risk control while trying to incrementally "
            "improve risk-adjusted returns."
        )

        user_prompt = (
            "You will receive the current system state as JSON.\n\n"
            "Goals:\n"
            "1) Suggest SMALL, incremental changes to numeric parameters only.\n"
            "2) Keep risk under control; never recommend reckless leverage.\n"
            "3) NEVER enable short selling (allow_short_selling must remain false).\n"
            "4) Provide short commentary on performance and risk.\n\n"
            "Return ONLY a JSON object with the following keys:\n"
            "  strategy_updates: list of objects {\"name\": str, "
            "\"params\": {str: number}}\n"
            "  risk_updates: object of numeric fields to tweak "
            "(e.g. max_notional_per_order, max_cash_utilization)\n"
            "  overall_assessment: string\n"
            "  detailed_recommendations: object with keys 'risk', "
            "'strategies', 'operations' (all strings)\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {
                "role": "user",
                "content": f"Current state JSON (stringified):\n{state_json}",
            },
        ]

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                # DO NOT send temperature; some models reject it
                max_completion_tokens=self.max_output_tokens,
            )
        except Exception as exc:
            logger.warning(
                "AIController: OpenAI request failed: %s", exc, exc_info=True
            )
            return {}

        msg = resp.choices[0].message
        raw_content = getattr(msg, "content", None)

        # Handle both string and list-of-parts shapes.
        if isinstance(raw_content, list):
            parts: List[str] = []
            for part in raw_content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_obj = part.get("text")
                        if isinstance(text_obj, dict):
                            value = text_obj.get("value")
                            if isinstance(value, str):
                                parts.append(value)
                        elif isinstance(text_obj, str):
                            parts.append(text_obj)
                    else:
                        parts.append(str(part))
                elif isinstance(part, str):
                    parts.append(part)
            content_str = "\n".join(parts).strip()
        elif isinstance(raw_content, str):
            content_str = raw_content.strip()
        else:
            content_str = ("" if raw_content is None else str(raw_content)).strip()

        if not content_str:
            logger.warning(
                "AIController: empty content in model response; raw message=%r",
                msg,
            )
            return {}

        # Try to parse JSON; if the model wrapped it with text, strip around it.
        def _parse_json(text: str) -> Optional[Mapping[str, Any]]:
            try:
                parsed_local = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("{")
                end = text.rfind("}")
                if start == -1 or end <= start:
                    return None
                try:
                    parsed_local = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None
            if not isinstance(parsed_local, Mapping):
                return None
            return parsed_local

        parsed = _parse_json(content_str)
        if parsed is None:
            logger.warning(
                "AIController: model returned non-JSON or badly formatted JSON; "
                "first 200 chars: %r",
                content_str[:200],
            )
            return {}

        return parsed

    # ------------------------------------------------------------------ #
    # Apply updates
    # ------------------------------------------------------------------ #

    def _apply_response(
        self,
        response: Mapping[str, Any],
        risk_config: Any,
        strategies: List[Any],
    ) -> None:
        overall = response.get("overall_assessment")
        if isinstance(overall, str) and overall:
            logger.info("AI overall assessment:\n%s", overall)

        detailed = response.get("detailed_recommendations") or {}
        if isinstance(detailed, Mapping):
            for key in ("risk", "strategies", "operations"):
                text = detailed.get(key)
                if isinstance(text, str) and text:
                    logger.info("AI recommendations (%s):\n%s", key, text)

        strat_updates = response.get("strategy_updates") or []
        risk_updates = response.get("risk_updates") or {}

        if strat_updates:
            logger.info("AI suggested strategy_updates: %s", strat_updates)
            self._apply_strategy_updates(strat_updates, strategies)

        if risk_updates:
            logger.info("AI suggested risk_updates: %s", risk_updates)
            self._apply_risk_updates(risk_updates, risk_config)

    def _apply_strategy_updates(
        self,
        updates: List[Mapping[str, Any]],
        strategies: List[Any],
    ) -> None:
        by_name: Dict[Optional[str], Any] = {
            getattr(s, "name", None): s for s in strategies
        }

        for upd in updates:
            if not isinstance(upd, Mapping):
                continue

            name = upd.get("name")
            params = upd.get("params") or {}
            strat = by_name.get(name)
            if strat is None:
                logger.debug(
                    "AIController: no strategy with name=%r; skipping strategy update",
                    name,
                )
                continue

            for key, val in params.items():
                if not hasattr(strat, key):
                    logger.debug(
                        "AIController: strategy %r has no attribute %r; skipping",
                        name,
                        key,
                    )
                    continue

                old = getattr(strat, key)
                if not isinstance(old, (int, float)):
                    logger.debug(
                        "AIController: strategy %r attribute %r is not numeric; skipping",
                        name,
                        key,
                    )
                    continue

                if not isinstance(val, (int, float)):
                    logger.debug(
                        "AIController: suggested value for %r.%r is not numeric; skipping",
                        name,
                        key,
                    )
                    continue

                new_val = float(val)

                # Clamp change: at most 3x change in magnitude.
                if old != 0:
                    ratio = abs(new_val / old)
                    if ratio > 3.0:
                        new_val = old * (3.0 if new_val > 0 else -3.0)

                setattr(strat, key, new_val)
                logger.info(
                    "AI updated strategy %s: %s %.4f -> %.4f",
                    name,
                    key,
                    old,
                    new_val,
                )

    def _apply_risk_updates(
        self,
        updates: Mapping[str, Any],
        risk_config: Any,
    ) -> None:
        for key, val in updates.items():
            if not hasattr(risk_config, key):
                logger.debug(
                    "AIController: risk_config has no attribute %r; skipping", key
                )
                continue

            old = getattr(risk_config, key)
            if isinstance(old, bool):
                # Never allow shorts, even if the model tries.
                if key == "allow_short_selling":
                    logger.info(
                        "AIController: ignoring suggestion to change allow_short_selling"
                    )
                    continue

                new_bool = bool(val)
                setattr(risk_config, key, new_bool)
                logger.info(
                    "AI updated risk_config bool: %s %r -> %r", key, old, new_bool
                )
                continue

            if not isinstance(old, (int, float)):
                logger.debug(
                    "AIController: risk_config attribute %r is not numeric/bool; skipping",
                    key,
                )
                continue

            if not isinstance(val, (int, float)):
                logger.debug(
                    "AIController: suggested value for risk_config.%s is not numeric; skipping",
                    key,
                )
                continue

            new_val = float(val)

            # Clamp change to at most 2x.
            if old != 0:
                ratio = abs(new_val / old)
                if ratio > 2.0:
                    new_val = old * (2.0 if new_val > 0 else -2.0)

            setattr(risk_config, key, new_val)
            logger.info(
                "AI updated risk_config: %s %.4f -> %.4f", key, old, new_val
            )

        # Final safety: ensure allow_short_selling is False.
        if getattr(risk_config, "allow_short_selling", False):
            setattr(risk_config, "allow_short_selling", False)
            logger.info(
                "AIController enforced allow_short_selling=False for safety."
            )
