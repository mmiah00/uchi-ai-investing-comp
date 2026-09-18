"""IBKR data access layer.

Wraps ib_insync to pull three things for a ticker from TWS / IB Gateway:

1. Daily historical bars (for fitting the GARCH volatility model)
2. A live snapshot quote (last/bid/ask, for "where are we right now")
3. Fundamental data (Reuters snapshot + ratios + analyst estimates)

Requires TWS or IB Gateway running locally with API access enabled
(Configure > API > Settings > Enable ActiveX and Socket Clients), and
"Read-Only API" is fine since this module never places orders.

Fundamentals are fetched and returned as a flat dict today. They are not
yet fed into the GARCH mean/variance equations -- see garch_model.py for
where that hook goes when we add more variables later.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
from ib_insync import IB, Stock, util

from .config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    symbol: str
    last: float | None
    bid: float | None
    ask: float | None
    close: float | None
    timestamp: datetime


class IBKRClient:
    """Thin, purpose-built wrapper around ib_insync.IB for this pipeline."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.ib = IB()

    def __enter__(self) -> "IBKRClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def connect(self) -> None:
        ib_cfg = self.cfg.ibkr
        logger.info("Connecting to IBKR at %s:%s (clientId=%s)", ib_cfg.host, ib_cfg.port, ib_cfg.client_id)
        self.ib.connect(ib_cfg.host, ib_cfg.port, clientId=ib_cfg.client_id, timeout=ib_cfg.timeout)

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def _contract(self) -> Stock:
        c = self.cfg.contract
        contract = Stock(c.symbol, c.exchange, c.currency, primaryExchange=c.primary_exchange)
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f"Could not qualify contract for {c.symbol!r} on IBKR")
        return qualified[0]

    def fetch_historical_bars(self) -> pd.DataFrame:
        """Daily OHLCV bars going back `data.lookback_years` years."""
        contract = self._contract()
        duration = f"{self.cfg.data.lookback_years} Y"
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=self.cfg.data.bar_size,
            whatToShow=self.cfg.data.what_to_show,
            useRTH=self.cfg.data.use_rth,
            formatDate=1,
        )
        if not bars:
            raise RuntimeError(
                f"IBKR returned no historical bars for {self.cfg.contract.symbol}. "
                "Check market data subscriptions and that the contract is correct."
            )
        df = util.df(bars)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_snapshot(self) -> MarketSnapshot:
        """One-shot live snapshot quote (requires live/delayed market data permission)."""
        contract = self._contract()
        ticker = self.ib.reqMktData(contract, "", snapshot=True)
        self.ib.sleep(2.5)  # give IBKR time to populate the snapshot fields
        snap = MarketSnapshot(
            symbol=self.cfg.contract.symbol,
            last=_clean(ticker.last),
            bid=_clean(ticker.bid),
            ask=_clean(ticker.ask),
            close=_clean(ticker.close),
            timestamp=datetime.now(),
        )
        self.ib.cancelMktData(contract)
        return snap

    def fetch_fundamentals(self) -> dict:
        """Pulls Reuters snapshot, ratios, and analyst estimates reports and
        flattens the useful fields into a single dict.

        These are stored for later use (e.g. as GARCH-X regressors or to
        inform the drift assumption in the simulation) but are not yet
        consumed by the model -- see pipeline.py TODO.
        """
        contract = self._contract()
        fundamentals: dict = {}

        for report_type, parser in (
            ("ReportSnapshot", _parse_report_snapshot),
            ("ReportRatios", _parse_report_ratios),
            ("RESC", _parse_analyst_estimates),
        ):
            try:
                xml_str = self.ib.reqFundamentalData(contract, report_type)
                if xml_str:
                    fundamentals.update(parser(xml_str))
            except Exception:
                logger.exception("Failed to fetch/parse fundamentals report %s", report_type)

        return fundamentals


def _clean(value: float) -> float | None:
    import math

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _parse_report_snapshot(xml_str: str) -> dict:
    """Pulls headline valuation/growth ratios out of the Reuters company snapshot report."""
    out = {}
    root = ET.fromstring(xml_str)
    for ratio in root.iter("Ratio"):
        field_name = ratio.get("FieldName")
        text = ratio.text
        if field_name and text:
            out[f"snapshot_{field_name}"] = _to_number(text)
    return out


def _parse_report_ratios(xml_str: str) -> dict:
    """Pulls the group/ratio table (valuation, growth, profitability, financial strength)."""
    out = {}
    root = ET.fromstring(xml_str)
    for group in root.iter("Group"):
        group_id = group.get("ID", "")
        for ratio in group.iter("Ratio"):
            field_id = ratio.get("FieldName", ratio.get("ID", ""))
            text = ratio.text
            if field_id and text:
                out[f"ratio_{group_id}_{field_id}"] = _to_number(text)
    return out


def _parse_analyst_estimates(xml_str: str) -> dict:
    """Pulls consensus analyst EPS/revenue estimates (RESC report)."""
    out = {}
    root = ET.fromstring(xml_str)
    for element in root.iter("ConsEstimate"):
        estimate_type = element.get("type", "")
        for period in element.iter("FYEstimate"):
            key = f"estimate_{estimate_type}_{period.get('relPeriod', '')}"
            if period.text:
                out[key] = _to_number(period.text)
    return out


def _to_number(text: str) -> float | str:
    try:
        return float(text)
    except (TypeError, ValueError):
        return text
