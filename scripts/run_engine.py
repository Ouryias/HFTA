# scripts/run_engine.py

from __future__ import annotations

import logging

from HFTA.broker.client import WealthsimpleClient
from HFTA.core.engine import Engine
from HFTA.core.order_manager import OrderManager
from HFTA.core.risk_manager import RiskConfig, RiskManager
from HFTA.strategies.micro_market_maker import MicroMarketMaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


def main() -> None:
    client = WealthsimpleClient()

    # Strategy config (still very small size)
    mm = MicroMarketMaker(
        name="mm_AAPL",
        config={
            "symbol": "AAPL",
            "max_inventory": 2,
            "spread": 0.05,         # 5 cents around mid
            "order_quantity": 1,
        },
    )

    # Risk config for this test:
    # - Max $50 per order
    # - At most 10% of available cash per BUY
    risk_cfg = RiskConfig(
        max_notional_per_order=50.0,
        max_cash_utilization=0.10,
    )
    risk_manager = RiskManager(risk_cfg)

    # Order manager, still dry-run (live=False)
    order_manager = OrderManager(
        client=client,
        risk_manager=risk_manager,
        live=False,               # flip to True only when you're ready
    )

    engine = Engine(
        client=client,
        strategies=[mm],
        symbols=["AAPL"],
        order_manager=order_manager,
        poll_interval=5.0,
    )

    print("Starting HFTA engine in DRY-RUN mode (with basic risk). Ctrl+C to stop.")
    engine.run_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
