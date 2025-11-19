# HFTA/market/models.py

from __future__ import annotations

from datetime import datetime

from peewee import (
    Model,
    CharField,
    DateTimeField,
    FloatField,
    AutoField,
    CompositeKey,
)

from HFTA.market.db import db


class BaseModel(Model):
    class Meta:
        database = db


class QuoteBar(BaseModel):
    """
    One snapshot per quote event.

    Later you can build daily OHLC from this.
    """
    id = AutoField()
    symbol = CharField(max_length=16)
    ts = DateTimeField(default=datetime.utcnow, index=True)
    last = FloatField(null=True)
    bid = FloatField(null=True)
    ask = FloatField(null=True)
    source = CharField(max_length=32, default="wealthsimple")

    class Meta:
        indexes = (
            (("symbol", "ts"), False),
        )
