# scripts/test_backtester.py

from HFTA.sim.backtester import BacktestConfig, BacktestEngine
from HFTA.strategies.micro_market_maker import MicroMarketMaker
from HFTA.core.risk_manager import RiskConfig


def main() -> None:
    # Allow the strategy to actually trade
    risk_cfg = RiskConfig(
        max_notional_per_order=10_000.0,   # allow orders up to $10k
        max_cash_utilization=0.5,          # a single BUY can use up to 50% of cash
        allow_short_selling=False,         # still no shorts
    )

    cfg = BacktestConfig(
        symbol="AAPL",
        starting_price=270.0,
        starting_cash=100_000.0,
        steps=200,
        step_seconds=5,
        volatility_annual=0.4,
        spread_cents=0.10,
        risk_config=risk_cfg,
    )

    mm_cfg = {
        "symbol": "AAPL",
        "max_inventory": 5,
        "spread": 0.10,
        "order_quantity": 1,
    }
    strat = MicroMarketMaker(name="mm_AAPL", config=mm_cfg)

    engine = BacktestEngine(strategies=[strat], config=cfg)
    result = engine.run()

    print("=== TEST BACKTEST RUN ===")
    print(f"Symbol: {result.symbol}")
    print(f"Starting cash: {result.starting_cash:,.2f}")
    print(f"Final cash: {result.final_cash:,.2f}")
    print(f"Final equity: {result.final_equity:,.2f}")
    print(f"Realized PnL: {result.realized_pnl:,.2f}")
    print(f"Max drawdown: {result.max_drawdown:.2%}")
    print(f"Equity points: {len(result.equity_curve)}")
    print(f"Positions at end: {result.positions_summary}")


if __name__ == "__main__":
    main()
