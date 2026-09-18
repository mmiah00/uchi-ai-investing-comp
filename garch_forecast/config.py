"""Central configuration for the FN GARCH forecasting pipeline.

Everything that is likely to change between runs (ticker, IBKR connection
details, model knobs) lives here so the rest of the codebase can stay free
of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IBKRConfig:
    """Connection details for TWS / IB Gateway.

    Defaults match the standard paper-trading TWS port. Live TWS uses 7496,
    live Gateway uses 4001, paper Gateway uses 4002.

    host defaults to the Windows host IP rather than 127.0.0.1 because this
    runs from WSL2 and TWS runs on the Windows side -- WSL2 is a separate
    network namespace, so localhost doesn't reach it. Find the current IP
    with `ipconfig` on Windows if it changes between reboots (NAT networking
    mode reassigns it; mirrored mode wouldn't need this override at all).
    """

    host: str = "172.27.224.1"
    port: int = 7497
    client_id: int = 17
    timeout: float = 15.0


@dataclass
class ContractConfig:
    symbol: str = "FN"
    sec_type: str = "STK"
    exchange: str = "SMART"
    primary_exchange: str = "NYSE"
    currency: str = "USD"


@dataclass
class DataConfig:
    # Years of daily history pulled from IBKR to estimate the GARCH process.
    # GARCH volatility estimation benefits from several years of data so the
    # model sees multiple vol regimes (not just the current calm/stormy one).
    lookback_years: int = 8
    bar_size: str = "1 day"
    what_to_show: str = "TRADES"
    use_rth: bool = True


@dataclass
class ModelConfig:
    # Mean model for the return equation. "Constant" is standard for GARCH
    # since GARCH's job is the variance equation, not return prediction.
    mean_model: str = "Constant"
    # GJR-GARCH captures the leverage effect (vol rises more after negative
    # returns than positive ones), which is well documented for single
    # stocks. Falls back to plain GARCH if p/o/q below are set accordingly.
    vol_model: str = "GARCH"
    p: int = 1
    o: int = 1  # asymmetry (GJR) term; set to 0 for symmetric GARCH
    q: int = 1
    dist: str = "t"  # Student-t innovations to capture fat tails
    rescale: bool = True

    # Forecast horizon in trading days. ~21 trading days/month.
    horizon_months: tuple = (3, 6, 12)
    trading_days_per_month: int = 21

    # Monte Carlo simulation paths for the horizon price distribution.
    n_simulations: int = 20_000
    random_seed: int = 42

    # Confidence bands reported in the output.
    quantiles: tuple = (0.05, 0.25, 0.5, 0.75, 0.95)


@dataclass
class AppConfig:
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cache_dir: str = "cache"
    output_dir: str = "output"
