"""End-to-end orchestration: IBKR data -> GARCH fit -> Monte Carlo forecast -> report.

Usage:
    python -m garch_forecast.pipeline                      # live IBKR pull
    python -m garch_forecast.pipeline --use-cache           # reuse last cached pull
    python -m garch_forecast.pipeline --port 4002           # paper Gateway instead of paper TWS
    python -m garch_forecast.pipeline --synthetic           # no IBKR needed, demo/dev only

TWS or IB Gateway must be running and logged in, with API access enabled
(File > Global Configuration > API > Settings) before a live/--use-cache-less
run will work. See README.md for setup notes.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AppConfig
from .garch_model import (
    compute_log_returns,
    fit_garch,
    simulate_price_paths,
    summarize_horizons,
    summarize_horizons_to_frame,
)
from .ibkr_data import IBKRClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Sequential blue ramp + chrome from the project's validated palette (see
# garch_forecast/README section on the fan chart for where these come from).
PALETTE = {
    "surface": "#fcfcfb",
    "primary_ink": "#0b0b0b",
    "secondary_ink": "#52514e",
    "muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
    "band_outer": "#cde2fb",  # step 100
    "band_mid": "#6da7ec",  # step 300
    "band_inner": "#256abf",  # step 500
    "median_line": "#184f95",  # step 600
    "history_line": "#52514e",  # secondary ink, so it doesn't compete with the forecast hue
}


def bars_cache_path(cfg: AppConfig) -> Path:
    return Path(cfg.cache_dir) / f"{cfg.contract.symbol}_daily_bars.csv"


def fundamentals_cache_path(cfg: AppConfig) -> Path:
    return Path(cfg.cache_dir) / f"{cfg.contract.symbol}_fundamentals.json"


def load_data(cfg: AppConfig, use_cache: bool, synthetic: bool) -> tuple[pd.DataFrame, dict, float | None]:
    """Returns (historical daily bars, fundamentals dict, live snapshot last price or None)."""
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    if synthetic:
        logger.warning("Running with --synthetic data. This is NOT real FN data; for pipeline dev/testing only.")
        return _synthetic_bars(cfg), {}, None

    bars_path = bars_cache_path(cfg)
    fund_path = fundamentals_cache_path(cfg)

    if use_cache:
        if not bars_path.exists():
            raise FileNotFoundError(f"--use-cache requested but no cache at {bars_path}. Run once without it first.")
        bars = pd.read_csv(bars_path, index_col="date", parse_dates=True)
        fundamentals = json.loads(fund_path.read_text()) if fund_path.exists() else {}
        return bars, fundamentals, None

    with IBKRClient(cfg) as client:
        bars = client.fetch_historical_bars()
        fundamentals = client.fetch_fundamentals()
        try:
            snapshot = client.fetch_snapshot()
            live_price = snapshot.last or snapshot.close
        except Exception:
            logger.exception("Live snapshot fetch failed; will fall back to last historical close")
            live_price = None

    bars.to_csv(bars_path, index_label="date")
    fund_path.write_text(json.dumps(fundamentals, indent=2, default=str))
    return bars, fundamentals, live_price


def _synthetic_bars(cfg: AppConfig) -> pd.DataFrame:
    """GBM-with-vol-clustering stand-in so the rest of the pipeline is runnable without IBKR."""
    rng = np.random.default_rng(cfg.model.random_seed)
    n_days = cfg.data.lookback_years * 252
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)

    daily_vol = np.full(n_days, 0.018)
    t_df = 5
    shocks = rng.standard_t(df=t_df, size=n_days) / np.sqrt(t_df / (t_df - 2))  # unit-variance so the vol recursion stays stationary
    for t in range(1, n_days):
        daily_vol[t] = np.sqrt(0.000002 + 0.08 * (daily_vol[t - 1] * shocks[t - 1]) ** 2 + 0.90 * daily_vol[t - 1] ** 2)
    returns = 0.0003 + daily_vol * shocks
    prices = 90.0 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"close": prices}, index=dates)
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["high"] = df[["open", "close"]].max(axis=1) * 1.005
    df["low"] = df[["open", "close"]].min(axis=1) * 0.995
    df["volume"] = 1_000_000
    df.index.name = "date"
    return df


def run(cfg: AppConfig, use_cache: bool = False, synthetic: bool = False, plot: bool = True) -> pd.DataFrame:
    bars, fundamentals, live_price = load_data(cfg, use_cache=use_cache, synthetic=synthetic)
    logger.info("Loaded %d daily bars, %d fundamentals fields", len(bars), len(fundamentals))

    last_price = live_price if live_price is not None else float(bars["close"].iloc[-1])
    logger.info("Anchoring simulation at last price: %.2f (%s)", last_price, "live" if live_price else "last close")

    log_returns = compute_log_returns(bars)
    fit_result = fit_garch(log_returns, cfg.model)
    logger.info("\n%s", fit_result.summary())

    price_paths = simulate_price_paths(fit_result, last_price, cfg.model)
    horizons = summarize_horizons(price_paths, last_price, fit_result, cfg.model)
    summary_df = summarize_horizons_to_frame(horizons)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{cfg.contract.symbol}_garch_forecast.csv"
    summary_df.to_csv(summary_path)
    logger.info("Wrote forecast summary to %s", summary_path)

    if fundamentals:
        (out_dir / f"{cfg.contract.symbol}_fundamentals.json").write_text(
            json.dumps(fundamentals, indent=2, default=str)
        )

    if plot:
        plot_path = out_dir / f"{cfg.contract.symbol}_forecast_fan_chart.png"
        _plot_fan_chart(bars, price_paths, last_price, cfg, plot_path)
        logger.info("Wrote fan chart to %s", plot_path)

    return summary_df


def _plot_fan_chart(bars: pd.DataFrame, price_paths: np.ndarray, last_price: float, cfg: AppConfig, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=PALETTE["surface"])
    ax.set_facecolor(PALETTE["surface"])

    # Trailing 12 months of actual closes for context.
    history = bars["close"].iloc[-252:]
    ax.plot(history.index, history.values, color=PALETTE["history_line"], linewidth=1.5, label="Historical close")

    horizon_days = price_paths.shape[1]
    future_dates = pd.bdate_range(start=history.index[-1], periods=horizon_days + 1)[1:]

    q05 = np.quantile(price_paths, 0.05, axis=0)
    q25 = np.quantile(price_paths, 0.25, axis=0)
    q50 = np.quantile(price_paths, 0.50, axis=0)
    q75 = np.quantile(price_paths, 0.75, axis=0)
    q95 = np.quantile(price_paths, 0.95, axis=0)

    ax.fill_between(future_dates, q05, q95, color=PALETTE["band_outer"], linewidth=0, label="5th-95th pct")
    ax.fill_between(future_dates, q25, q75, color=PALETTE["band_mid"], linewidth=0, label="25th-75th pct")
    ax.plot(future_dates, q50, color=PALETTE["median_line"], linewidth=2, label="Median forecast")

    ax.axvline(history.index[-1], color=PALETTE["baseline"], linewidth=1, linestyle="--")
    ax.scatter([history.index[-1]], [last_price], color=PALETTE["primary_ink"], zorder=5, s=25)

    ax.set_title(f"{cfg.contract.symbol} — GARCH Monte Carlo forecast", color=PALETTE["primary_ink"], fontsize=13, loc="left")
    ax.set_ylabel("Price (USD)", color=PALETTE["secondary_ink"])
    ax.tick_params(colors=PALETTE["muted"])
    ax.grid(True, color=PALETTE["gridline"], linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["baseline"])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    legend = ax.legend(loc="upper left", frameon=False, labelcolor=PALETTE["secondary_ink"])
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=PALETTE["surface"])
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="FN")
    parser.add_argument("--host", default="172.27.224.1", help="TWS/Gateway host; the Windows host IP when running from WSL2")
    parser.add_argument("--port", type=int, default=7497, help="7497=paper TWS, 7496=live TWS, 4002=paper Gateway, 4001=live Gateway")
    parser.add_argument("--client-id", type=int, default=17)
    parser.add_argument("--lookback-years", type=int, default=8)
    parser.add_argument("--n-simulations", type=int, default=20_000)
    parser.add_argument("--use-cache", action="store_true", help="Skip IBKR and reuse the last cached data pull")
    parser.add_argument("--synthetic", action="store_true", help="Skip IBKR and use synthetic data (dev/testing only)")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = AppConfig()
    cfg.contract.symbol = args.symbol
    cfg.ibkr.host = args.host
    cfg.ibkr.port = args.port
    cfg.ibkr.client_id = args.client_id
    cfg.data.lookback_years = args.lookback_years
    cfg.model.n_simulations = args.n_simulations

    summary_df = run(cfg, use_cache=args.use_cache, synthetic=args.synthetic, plot=not args.no_plot)
    print(summary_df.to_string())


if __name__ == "__main__":
    main()
