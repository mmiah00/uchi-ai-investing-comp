"""Builds the FN dashboard (web/template.html + market data -> output/FN_dashboard.html).

This fetches real daily price history from Yahoo Finance's public chart
endpoint (no API key needed) rather than IBKR, because generating this
dashboard doesn't require a local TWS/Gateway session the way
`garch_forecast.pipeline` does. Swap `_fetch_price_history` for
`IBKRClient.fetch_historical_bars` once you want this sourced from the same
live feed as the rest of the pipeline.

The FUNDAMENTALS dict below is a hand-entered snapshot (sourced from public
filings/aggregators, see the comment above it) -- IBKR's fundamentals XML
field names haven't been mapped to this schema yet since that requires a
live connection to inspect the actual payload. See ibkr_data.py:fetch_fundamentals
for the raw pull; wiring it into FUNDAMENTALS below is the next step once
TWS/Gateway is reachable.

Usage:
    python3 web/build_dashboard_data.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from garch_forecast.config import AppConfig
from garch_forecast.garch_model import compute_log_returns, fit_garch, simulate_price_paths, daily_quantile_bands

SYMBOL = "FN"
EPS_TTM = 13.05  # holds constant across the P/E trend chart; resets quarterly, not daily

# Hand-entered from stockanalysis.com / gurufocus.com, accessed 2026-09-18.
# Replace with a live IBKR fundamentals pull (see ibkr_data.py) when available.
FUNDAMENTALS = {
    "marketCap": 14.02e9,
    "sharesOut": 35.83e6,
    "peTtm": 29.98,
    "peFwd": 21.55,
    "epsTtm": EPS_TTM,
    "revenueTtm": 4.64e9,
    "revenueGrowthYoY": 0.357,
    "netIncomeTtm": 473.03e6,
    "netIncomeGrowthYoY": 0.423,
    "roe": 0.1999,
    "roic": 0.3023,
    "beta": 1.21,
    "week52Low": 354.41,
    "week52High": 748.89,
    "analystRating": "Buy",
    "analystCount": 9,
    "priceTarget": 734.11,
}


def _fetch_price_history(symbol: str, range_: str = "8y") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read())
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    close = result["indicators"]["quote"][0]["close"]
    dates = (
        pd.to_datetime(ts, unit="s")
        .tz_localize("UTC")
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None)
    )
    bars = pd.DataFrame({"close": close}, index=dates)
    bars = bars[~bars.index.duplicated(keep="first")].dropna().sort_index()
    bars.index.name = "date"
    return bars


def build_data_payload(bars: pd.DataFrame, cfg: AppConfig) -> dict:
    last_price = float(bars["close"].iloc[-1])
    last_date = bars.index[-1]

    log_returns = compute_log_returns(bars)
    fit_result = fit_garch(log_returns, cfg.model)

    max_months = max(cfg.model.horizon_months)
    horizon_days = max_months * cfg.model.trading_days_per_month
    price_paths = simulate_price_paths(fit_result, last_price, cfg.model)
    bands = daily_quantile_bands(price_paths, cfg.model.quantiles)
    future_dates = pd.bdate_range(start=last_date, periods=horizon_days + 1)[1:]

    history = bars["close"].iloc[-64:]
    pe_window = bars["close"].iloc[-22:]

    h12 = horizon_days - 1
    log_ret_12m = np.log(price_paths[:, h12] / last_price)

    return {
        "generated_at": pd.Timestamp.now().isoformat(),
        "as_of_date": str(last_date.date()),
        "last_price": round(last_price, 2),
        "day_change": {
            "abs": round(float(bars["close"].iloc[-1] - bars["close"].iloc[-2]), 2),
            "pct": round(float(bars["close"].iloc[-1] / bars["close"].iloc[-2] - 1), 4),
        },
        "history": [{"date": str(d.date()), "close": round(float(v), 2)} for d, v in history.items()],
        "pe_trend": [
            {"date": str(d.date()), "pe": round(float(v) / EPS_TTM, 2)} for d, v in pe_window.items()
        ],
        "forecast": [
            {
                "date": str(d.date()),
                "q05": round(float(bands[0.05][i]), 2),
                "q25": round(float(bands[0.25][i]), 2),
                "q50": round(float(bands[0.5][i]), 2),
                "q75": round(float(bands[0.75][i]), 2),
                "q95": round(float(bands[0.95][i]), 2),
            }
            for i, d in enumerate(future_dates)
        ],
        "prob_positive_12m": round(float(np.mean(price_paths[:, h12] > last_price)), 4),
        "model": {
            "vol_model": "GJR-GARCH(1,1,1)",
            "dist": "Student-t",
            "nu": float(fit_result.params.get("nu", float("nan"))),
            "n_observations": int(fit_result.nobs),
            "n_simulations": cfg.model.n_simulations,
            "annualized_vol_12m": round(float(np.std(log_ret_12m)), 4),
        },
    }


def render_dashboard(data_payload: dict, fundamentals: dict, template_path: Path, out_path: Path) -> None:
    template = template_path.read_text()
    combined = dict(data_payload)
    combined["fundamentals"] = fundamentals

    review_path = REPO_ROOT / "output" / "FN_short_thesis_review.json"
    if review_path.exists():
        combined["review"] = json.loads(review_path.read_text())

    injected = template.replace("__FN_DATA_JSON__", json.dumps(combined, separators=(",", ":")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(injected)


def main() -> None:
    cfg = AppConfig()
    cfg.contract.symbol = SYMBOL
    cfg.model.n_simulations = 20_000

    bars = _fetch_price_history(SYMBOL)
    print(f"Fetched {len(bars)} daily bars for {SYMBOL}, {bars.index.min().date()} - {bars.index.max().date()}")

    payload = build_data_payload(bars, cfg)
    out_path = REPO_ROOT / "output" / f"{SYMBOL}_dashboard.html"
    render_dashboard(payload, FUNDAMENTALS, REPO_ROOT / "web" / "template.html", out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
