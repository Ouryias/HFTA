# scripts/init_db.py

from __future__ import annotations

from HFTA.market.db import db
from HFTA.market.models import QuoteBar


def main() -> None:
    db.connect(reuse_if_open=True)
    db.create_tables([QuoteBar])
    print("DB init complete. Tables created: QuoteBar")


if __name__ == "__main__":
    main()
