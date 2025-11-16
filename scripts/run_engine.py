# scripts/run_engine.py

from __future__ import annotations

import logging

from HFTA.broker.client import WealthsimpleClient
from HFTA.core.engine import Engine
from HFTA.strategies.micro_market_maker import MicroMarketMaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)


def main() -> None:
    client = WealthsimpleClient()

    mm = MicroMarketMaker(
        name="mm_AAPL",
        config={
            "symbol": "AAPL",
            "max_inventory": 2,
            "spread": 0.05,         # 5 cents around mid
            "order_quantity": 1,
        },
    )

    engine = Engine(
        client=client,
        strategies=[mm],
        symbols=["AAPL"],
        poll_interval=5.0,
        live=False,                # DRY-RUN: no real orders yet
    )

    print("Starting HFTA engine in DRY-RUN mode. Ctrl+C to stop.")
    engine.run_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Extra safety in case anything escapes the engine-level handler
        print("\nStopped by user.")
