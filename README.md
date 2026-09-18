# uchi-ai-investing-comp

## FN GARCH forecast

A GARCH-based volatility/return forecasting pipeline for Fabrinet (FN), using
live/historical data pulled from Interactive Brokers (IBKR). It fits a
GJR-GARCH(1,1) model to daily log returns, then runs a Monte Carlo simulation
of the fitted process to produce a price/return distribution at 3, 6, and 12
month horizons.

Fundamentals are pulled from IBKR alongside the price history and cached, but
are not yet wired into the model — they're the first thing to add as an
additional variable (see `garch_forecast/garch_model.py` for the `drift_overrides`
hook where fundamentals-driven drift should plug in).

### Setup

```
pip install -r requirements.txt
```

You need TWS or IB Gateway running and logged in, with the API enabled:
File > Global Configuration > API > Settings > check "Enable ActiveX and
Socket Clients". A read-only API connection is sufficient — this pipeline
never places orders.

Default ports: `7497` paper TWS, `7496` live TWS, `4002` paper Gateway,
`4001` live Gateway.

### Run

```
python -m garch_forecast.pipeline                     # live pull from IBKR (paper TWS by default)
python -m garch_forecast.pipeline --port 4002          # paper Gateway
python -m garch_forecast.pipeline --use-cache          # reuse the last IBKR pull, no connection needed
python -m garch_forecast.pipeline --synthetic          # synthetic data, no IBKR needed (dev/testing only)
```

Outputs land in `output/`:
- `FN_garch_forecast.csv` — quantile price/return table per horizon
- `FN_forecast_fan_chart.png` — historical price + forecast fan chart
- `FN_fundamentals.json` — raw fundamentals pulled from IBKR, for future use

Raw IBKR pulls are cached in `cache/` so repeated runs/iteration don't need a
live connection every time (`--use-cache`).

### Model notes

- Mean equation: constant (GARCH's job is the variance equation, not
  predicting drift — see `ModelConfig` in `config.py`).
- Volatility equation: GJR-GARCH(1,1,1) with Student-t innovations, to
  capture both the leverage effect (vol rises more after down days) and
  fat tails.
- Horizon distribution comes from arch's simulation-based forecast (bootstrapped
  innovations consistent with the fitted variance process), compounded into
  price paths, not a closed-form formula — this is what lets the fan chart
  show the full distribution rather than just a point forecast.
