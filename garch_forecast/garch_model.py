"""GARCH-based return/volatility forecasting and Monte Carlo price simulation.

Why GARCH here: GARCH models the *volatility* of returns (its conditional
variance clusters and mean-reverts), not the price level directly. To turn
that into a "how is FN likely to perform 3-12 months out" answer we:

  1. Fit a GARCH(1,1)-type model (with an asymmetric/leverage term by
     default) to daily log returns to capture the current volatility
     regime and how it's expected to evolve/mean-revert.
  2. Use the fitted model's own return-innovation simulator to generate
     thousands of possible daily return paths out to the horizon.
  3. Compound each simulated path into a price path, then read off the
     distribution (median + confidence bands) at each requested horizon.

The mean (drift) of the return equation is currently a plain constant
estimated from historical returns -- appropriate for a first pass. The
`drift_overrides` hook in `simulate_price_paths` is where fundamentals
(e.g. analyst target price growth, valuation reversion, earnings growth)
should plug in once we add those variables: it lets you shift the daily
drift going forward without refitting the volatility process.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from .config import ModelConfig

logger = logging.getLogger(__name__)

RESCALE_FACTOR = 100.0  # returns are scaled to percent units before fitting; arch recommends this for optimizer stability


def compute_log_returns(prices: pd.DataFrame, price_col: str = "close") -> pd.Series:
    log_ret = np.log(prices[price_col]).diff().dropna()
    log_ret.name = "log_return"
    return log_ret


def fit_garch(returns: pd.Series, cfg: ModelConfig) -> ARCHModelResult:
    """Fits the configured GARCH variant to daily log returns.

    Returns are pre-scaled to percent units (x100) because arch's optimizer
    is numerically better behaved on that scale; RESCALE_FACTOR is divided
    back out everywhere downstream.
    """
    scaled_returns = returns * RESCALE_FACTOR
    am = arch_model(
        scaled_returns,
        mean=cfg.mean_model,
        vol=cfg.vol_model,
        p=cfg.p,
        o=cfg.o,
        q=cfg.q,
        dist=cfg.dist,
        rescale=False,  # already scaled manually above
    )
    result = am.fit(disp="off")
    logger.info("GARCH fit converged: %s", result.convergence_flag == 0)
    return result


@dataclass
class HorizonForecast:
    months: int
    trading_days: int
    price_quantiles: dict  # {quantile: price}
    return_quantiles: dict  # {quantile: cumulative simple return}
    prob_positive: float
    annualized_vol_forecast: float


def simulate_price_paths(
    fit_result: ARCHModelResult,
    last_price: float,
    cfg: ModelConfig,
    drift_overrides: np.ndarray | None = None,
) -> np.ndarray:
    """Simulates forward price paths using the fitted GARCH process.

    Uses arch's built-in simulation-based forecasting, which draws
    bootstrapped/parametric innovations consistent with the fitted
    variance process for each of `n_simulations` paths out to the max
    horizon requested.

    drift_overrides: optional array of length `horizon` (in percent, same
    scale as the fitted returns) to *add* to the simulated daily returns.
    This is the seam for injecting a fundamentals-driven drift later
    (e.g. distributing an analyst-implied annual return target across
    trading days) without touching the volatility model. Left None today.

    Returns an array of shape (n_simulations, horizon_days) of simulated
    price levels.
    """
    max_months = max(cfg.horizon_months)
    horizon_days = max_months * cfg.trading_days_per_month

    forecasts = fit_result.forecast(
        horizon=horizon_days,
        method="simulation",
        simulations=cfg.n_simulations,
        reindex=False,
    )
    # shape: (1, n_simulations, horizon_days) -> squeeze the single origin dim
    simulated_returns_pct = forecasts.simulations.values[0]  # percent units

    if drift_overrides is not None:
        if drift_overrides.shape[0] != horizon_days:
            raise ValueError("drift_overrides must have length == horizon_days")
        simulated_returns_pct = simulated_returns_pct + drift_overrides

    simulated_returns = simulated_returns_pct / RESCALE_FACTOR  # back to log-return decimal units
    cumulative_log_returns = np.cumsum(simulated_returns, axis=1)
    price_paths = last_price * np.exp(cumulative_log_returns)
    return price_paths


def summarize_horizons(
    price_paths: np.ndarray,
    last_price: float,
    fit_result: ARCHModelResult,
    cfg: ModelConfig,
) -> list[HorizonForecast]:
    """Reads off price/return quantiles and vol forecast at each requested horizon."""
    results = []
    for months in sorted(cfg.horizon_months):
        day_idx = months * cfg.trading_days_per_month - 1  # 0-indexed day within simulation
        prices_at_horizon = price_paths[:, day_idx]

        price_quantiles = {q: float(np.quantile(prices_at_horizon, q)) for q in cfg.quantiles}
        return_quantiles = {q: float(p / last_price - 1.0) for q, p in price_quantiles.items()}
        prob_positive = float(np.mean(prices_at_horizon > last_price))

        # Annualized vol implied by the simulated dispersion at this horizon.
        log_returns_at_horizon = np.log(prices_at_horizon / last_price)
        horizon_years = (day_idx + 1) / (cfg.trading_days_per_month * 12)
        annualized_vol = float(np.std(log_returns_at_horizon) / np.sqrt(horizon_years))

        results.append(
            HorizonForecast(
                months=months,
                trading_days=day_idx + 1,
                price_quantiles=price_quantiles,
                return_quantiles=return_quantiles,
                prob_positive=prob_positive,
                annualized_vol_forecast=annualized_vol,
            )
        )
    return results


def daily_quantile_bands(price_paths: np.ndarray, quantiles: tuple = (0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    """Full trading-day-by-trading-day quantile bands, for fan charts that plot every day
    rather than just the discrete 3/6/12-month markers `summarize_horizons` reads off."""
    return {q: np.quantile(price_paths, q, axis=0) for q in quantiles}


def summarize_horizons_to_frame(horizons: list[HorizonForecast]) -> pd.DataFrame:
    rows = []
    for h in horizons:
        row = {"months": h.months, "trading_days": h.trading_days, "prob_positive": h.prob_positive,
               "annualized_vol_forecast": h.annualized_vol_forecast}
        for q, price in h.price_quantiles.items():
            row[f"price_q{int(q * 100)}"] = price
        for q, ret in h.return_quantiles.items():
            row[f"return_q{int(q * 100)}"] = ret
        rows.append(row)
    return pd.DataFrame(rows).set_index("months")
