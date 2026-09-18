"""FRED macro data for the price-driver variables commonly cited for FN:
data center & AI infrastructure demand, and broader tech valuation/sentiment.

Customer concentration and company-level financial performance (revenue, margins,
EPS, buybacks) aren't macro series -- see fundamentals_edgar.py for those, pulled
straight from Fabrinet's SEC filings instead.

Requires a FRED API key (free, https://fred.stlouisfed.org/docs/api/api_key.html)
in a `.env` file at the repo root as `FRED-API-KEY=...`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# name -> (FRED series id, publication lag in days to shift the series by so a
# value is never visible before it was actually released -- avoids look-ahead
# bias when this gets forward-filled onto daily trading dates). Daily-frequency
# series publish same/next day so get a token 1-day lag; monthly/quarterly
# series lag by their typical release schedule.
DRIVER_SERIES = {
    # Data center / AI infrastructure demand proxies
    "it_capex": ("A679RC1Q027SBEA", 45),  # Private fixed investment: info processing equip & software (quarterly, BEA, ~6wk release lag)
    "electronics_production": ("IPG334S", 20),  # Industrial Production: Computer & Electronic Products (monthly, mid-month-plus release)
    "electronics_new_orders": ("A34SNO", 30),  # Manufacturers' New Orders: Computers & Electronic Products (monthly, ~1mo release lag)
    "semiconductor_ppi": ("PCU334413334413", 20),  # PPI: Semiconductor & Related Device Mfg (monthly, input-cost/margin proxy)
    # Valuation & broader tech sentiment
    "nasdaq": ("NASDAQCOM", 1),
    "vix": ("VIXCLS", 1),
    "treasury_10y": ("DGS10", 1),
    "credit_spread": ("BAA10Y", 1),
}


def _load_api_key(env_path: Path) -> str:
    values = dotenv_values(env_path)
    key = values.get("FRED-API-KEY")
    if not key:
        raise RuntimeError(f"FRED-API-KEY not found in {env_path}. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html")
    return key


def fetch_series(series_id: str, api_key: str, start: str = "2000-01-01") -> pd.Series:
    resp = requests.get(
        FRED_BASE,
        params={"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start},
        timeout=20,
    )
    resp.raise_for_status()
    obs = resp.json()["observations"]
    dates = pd.to_datetime([o["date"] for o in obs])
    values = pd.to_numeric([o["value"] for o in obs], errors="coerce")
    s = pd.Series(values, index=dates, name=series_id).dropna()
    return s


def fetch_driver_panel(env_path: Path, cache_dir: Path, start: str = "2000-01-01", use_cache: bool = False) -> pd.DataFrame:
    """Fetches every series in DRIVER_SERIES and returns them as named columns,
    each shifted forward by its publication lag but NOT yet forward-filled to a
    daily grid -- alignment onto trading days happens in drivers.py, where it's
    merged with price data.
    """
    cache_path = cache_dir / "fred_drivers.csv"
    if use_cache:
        if not cache_path.exists():
            raise FileNotFoundError(f"--use-cache requested but no cache at {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    api_key = _load_api_key(env_path)
    columns = {}
    for name, (series_id, lag_days) in DRIVER_SERIES.items():
        try:
            s = fetch_series(series_id, api_key, start=start)
            s.index = s.index + pd.Timedelta(days=lag_days)
            columns[name] = s
            logger.info("Fetched %s (%s): %d obs, publication-lagged %dd", name, series_id, len(s), lag_days)
        except Exception:
            logger.exception("Failed to fetch FRED series %s (%s)", name, series_id)

    panel = pd.DataFrame(columns)
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(cache_path, index_label="date")
    return panel
