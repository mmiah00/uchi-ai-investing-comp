"""Merges price data with the FRED macro panel and EDGAR fundamentals into one
daily feature panel, and fits an ARX-GJR-GARCH to see which of the commonly-cited
FN price drivers (data center demand, tech sentiment, financial performance) are
actually statistically significant for its daily returns.

This is deliberately kept separate from the forward 12-month Monte Carlo in
garch_model.py/pipeline.py: an exogenous-regressor forecast needs *future* values
of every regressor, and nobody has a credible 12-month path for the NASDAQ or VIX.
Treat this as the diagnostic that tells you which drivers matter and by how much;
`simulate_price_paths`'s `drift_overrides` hook is where a short-horizon version of
this (drivers held at their last known level) would plug into the forward sim.

All macro/fundamental series are shifted to when they actually became public
(FRED publication lag, EDGAR filing date) before being forward-filled onto trading
days and lagged one more day relative to the return being explained, so nothing
here is visible before the market could have seen it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from arch import arch_model

from . import fundamentals_edgar, macro_data
from .config import AppConfig
from .garch_model import RESCALE_FACTOR, compute_log_returns

logger = logging.getLogger(__name__)

# Transform applied to each raw column before it enters the regression: FRED
# levels get simple daily differences (rates, spreads) or daily pct-change
# (index levels), monthly/quarterly series get YoY pct-change to detrend them,
# and everything is then lagged 1 trading day relative to the return it explains.
DRIVER_TRANSFORMS = {
    "nasdaq": ("pct_change", 1),
    "vix": ("diff", 1),
    "treasury_10y": ("diff", 1),
    "credit_spread": ("diff", 1),
    "electronics_production": ("pct_change_yoy", 21),
    "electronics_new_orders": ("pct_change_yoy", 21),
    "it_capex": ("pct_change_yoy", 63),
    "semiconductor_ppi": ("pct_change_yoy", 21),
    "revenue_growth_yoy": ("level", 1),
    "eps_growth_yoy": ("level", 1),
}


def build_driver_panel(
    bars: pd.DataFrame,
    env_path: Path,
    cache_dir: Path,
    use_cache: bool = False,
    edgar_contact: str | None = None,
) -> pd.DataFrame:
    """Returns a DataFrame indexed on `bars`' trading days with columns: close,
    log_return, and one column per entry in DRIVER_TRANSFORMS, already transformed
    and lagged (see module docstring).
    """
    fred_panel = macro_data.fetch_driver_panel(env_path, cache_dir, use_cache=use_cache)

    edgar_cache = cache_dir / "edgar_fundamentals.csv"
    if use_cache:
        if not edgar_cache.exists():
            raise FileNotFoundError(f"--use-cache requested but no cache at {edgar_cache}")
        edgar_df = pd.read_csv(edgar_cache, index_col=0, parse_dates=True)
    else:
        edgar_q = fundamentals_edgar.build_quarterly_fundamentals(contact=edgar_contact)
        edgar_df = edgar_q[["revenue_growth_yoy", "eps_growth_yoy"]].copy()
        edgar_df.index = edgar_q["filed"]  # become public as of the filing date
        cache_dir.mkdir(parents=True, exist_ok=True)
        edgar_df.to_csv(edgar_cache, index_label="date")

    daily_index = bars.index
    raw = pd.DataFrame(index=daily_index)
    for col in fred_panel.columns:
        raw[col] = fred_panel[col].reindex(daily_index.union(fred_panel.index)).ffill().reindex(daily_index)
    for col in edgar_df.columns:
        raw[col] = edgar_df[col].reindex(daily_index.union(edgar_df.index)).ffill().reindex(daily_index)

    panel = pd.DataFrame(index=daily_index)
    panel["close"] = bars["close"]
    panel["log_return"] = compute_log_returns(bars).reindex(daily_index)

    for name, (kind, lag_days) in DRIVER_TRANSFORMS.items():
        if name not in raw.columns:
            continue
        s = raw[name]
        if kind == "pct_change":
            t = s.pct_change()
        elif kind == "diff":
            t = s.diff()
        elif kind == "pct_change_yoy":
            t = s.pct_change(252)  # ~1 trading year; series is already daily-ffilled
        elif kind == "level":
            t = s
        else:
            raise ValueError(f"unknown transform kind {kind!r}")
        panel[name] = t.shift(1)  # yesterday's known driver value explains today's return

    return panel


def fit_arx_garch(panel: pd.DataFrame, driver_cols: list[str] | None = None):
    """In-sample ARX-mean GJR-GARCH: explains daily log returns with the lagged
    driver panel, reusing the same GJR-GARCH(1,1,1) Student-t volatility spec as
    the main model. Returns the fitted result -- read `.summary()` or `.pvalues`
    to see which drivers are significant.
    """
    driver_cols = driver_cols or [c for c in panel.columns if c not in ("close", "log_return")]
    data = panel[["log_return"] + driver_cols].dropna()
    logger.info("ARX-GARCH fit sample: %d obs (%s to %s) after dropping rows with missing drivers", len(data), data.index.min().date(), data.index.max().date())

    y = data["log_return"] * RESCALE_FACTOR
    x = data[driver_cols]
    # Standardize regressors so coefficients are comparable in magnitude across
    # drivers on very different natural scales (bps vs. % vs. index points).
    x_std = (x - x.mean()) / x.std()

    am = arch_model(y, x=x_std, mean="LS", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)
    result = am.fit(disp="off")
    return result, x_std.columns.tolist()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars-cache", default="cache/FN_daily_bars.csv", help="Daily bars CSV from garch_forecast.pipeline (avoids a fresh IBKR pull)")
    parser.add_argument("--use-cache", action="store_true", help="Reuse cached FRED/EDGAR pulls instead of refetching")
    parser.add_argument("--edgar-contact", default=None, help="Email to include in the SEC EDGAR User-Agent header")
    args = parser.parse_args()

    bars_path = Path(args.bars_cache)
    if not bars_path.exists():
        raise FileNotFoundError(f"{bars_path} not found -- run `python -m garch_forecast.pipeline` first to populate it")
    bars = pd.read_csv(bars_path, index_col="date", parse_dates=True)

    cfg = AppConfig()
    panel = build_driver_panel(bars, Path(".env"), Path(cfg.cache_dir), use_cache=args.use_cache, edgar_contact=args.edgar_contact)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_dir / "FN_driver_panel.csv", index_label="date")

    result, driver_cols = fit_arx_garch(panel)
    print(result.summary())

    pvals = result.pvalues[driver_cols].sort_values()
    print("\nDrivers by significance (p-value):")
    print(pvals.to_string())

    (out_dir / "FN_driver_regression.txt").write_text(str(result.summary()))


if __name__ == "__main__":
    main()
