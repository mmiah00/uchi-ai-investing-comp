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

Running from WSL2 with TWS on the Windows side: `127.0.0.1` won't reach it
(separate network namespace) — `config.py` defaults to the Windows host IP
instead (`172.27.224.1` here). Find yours with `ipconfig` on Windows if it
changes between reboots, or override with `--host`.

Fundamentals require the Reuters Fundamentals data subscription on the
account (Client Portal > Settings > Market Data Subscriptions) — without it
`fetch_fundamentals` returns an empty dict rather than failing. Live quotes
fall back to delayed data automatically if the account isn't subscribed to
real-time for a symbol.

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

### Web dashboard

`web/` generates a standalone HTML dashboard (company summary, fundamentals
snapshot, and a daily 3-month-history + 12-month-forecast fan chart):

```
python3 web/build_dashboard_data.py
```

Writes `output/FN_dashboard.html` (open it directly in a browser). It pulls
price history from Yahoo Finance's public chart endpoint rather than IBKR,
since it doesn't require a local TWS/Gateway session the way
`garch_forecast.pipeline` does — swap in `IBKRClient.fetch_historical_bars`
there once you want it on the same live feed. The fundamentals shown are a
hand-entered snapshot (`FUNDAMENTALS` in `web/build_dashboard_data.py`,
sourced from public aggregators) — mapping IBKR's fundamentals XML fields
into that schema is the next step once TWS/Gateway is reachable to inspect
the real payload.

The dashboard leads with an **executive summary** making the short case
against FN, specifically vs. COHR and LITE (scenario price targets from
`SCENARIOS`, peer valuation from `PEER_VALUATION` — both in
`web/build_dashboard_data.py`, sourced from `FN_Short_Thesis_Model.xlsx` and
`FN_Short_Thesis_Report.docx`). It pulls in the peer margin chart,
EV/EBITDA reconciliation, and options-implied move from
`output/FN_short_thesis_review.json` when present — regenerate that via the
notebook (see below) before rebuilding the dashboard if the review data
looks stale.

### Short thesis model review

`notebooks/fn_short_thesis_review.ipynb` is a follow-up review of
`FN_Short_Thesis_Model.xlsx` (the standalone short-thesis workbook, not part of
`garch_forecast/`), covering six findings: no cached formula values (worked
around with the `formulas` library, since Excel/LibreOffice aren't available
here — still needs a real Ctrl+Alt+F9-and-save for the actual file), no peer
margin-degradation benchmark (real EDGAR data vs. COHR/LITE/CLS/JBL —
`garch_forecast/peer_comps.py`), no product-level margin breakdown (confirmed
unresolvable — FN reports one segment), an unreconciled EV/EBITDA discrepancy
(resolved exactly, plus a real formula bug found in the workbook), no quarterly
bridge to the Nov 2 earnings catalyst (built from FN's own quarterly EDGAR
history), and no options/positioning data (live IBKR options pull for the
implied move into earnings; open interest and short interest flagged as gaps,
not fabricated).

Run it with `jupyter nbconvert --to notebook --execute --inplace
notebooks/fn_short_thesis_review.ipynb` (needs IBKR connected for the last
section). It exports `output/FN_short_thesis_review.json`, which
`web/build_dashboard_data.py` picks up automatically and renders as a "Short
thesis review" section on the dashboard if present.

### Price drivers (FRED + SEC EDGAR)

`garch_forecast/drivers.py` builds a daily panel of the variables commonly cited
as FN price drivers -- data center/AI infrastructure demand, tech-sector
valuation/sentiment, and Fabrinet's own financial performance -- and fits an
ARX-GJR-GARCH to see which are actually statistically significant for its daily
returns:

```
python -m garch_forecast.drivers --use-cache   # reuse cached FRED/EDGAR/price pulls
python -m garch_forecast.drivers               # fresh pulls (needs .env + IBKR)
```

Needs a free FRED API key in `.env` at the repo root: `FRED-API-KEY=...` (get
one at https://fred.stlouisfed.org/docs/api/api_key.html). `.env` is gitignored.

Data sources per driver:
- **Data center / AI demand**: no FRED series tracks data center capex
  directly, so this uses proxies -- private fixed investment in information
  processing equipment (`A679RC1Q027SBEA`), industrial production and new
  orders for computer/electronic products (`IPG334S`, `A34SNO`), and
  semiconductor PPI (`PCU334413334413`) as a margin/input-cost signal.
- **Valuation & tech sentiment**: NASDAQ Composite, VIX, 10-year Treasury
  yield, and the Baa-10Y credit spread (`NASDAQCOM`, `VIXCLS`, `DGS10`,
  `BAA10Y`).
- **Financial performance**: real quarterly revenue/EPS growth and margins
  pulled straight from Fabrinet's 10-Q/10-K XBRL filings via SEC EDGAR's free
  API (`fundamentals_edgar.py`) -- no key needed, and a better source than
  IBKR's fundamentals endpoint, which this account isn't entitled to (see
  above).
- **Customer concentration**: not included as a regressor. Fabrinet discloses
  this qualitatively (reliance on a handful of large customers) in its 10-K
  risk factors, not as a structured time series -- there's nothing to pull.

Every series is shifted to when it actually became public (FRED publication
lag, EDGAR filing date) before being forward-filled onto trading days and
lagged one more day relative to the return it explains, so the regression
can't see anything before the market could have. As of the 2019-11 to
2026-09 sample (1,725 trading days), only two drivers clear p<0.10 at daily,
1-day-lagged frequency: `electronics_new_orders` (p≈0.056, positive — a
leading indicator of demand for what Fabrinet manufactures) and
`eps_growth_yoy` (p≈0.039, negative — reads as a valuation/mean-reversion
effect rather than "growth is bad"). Broad tech sentiment (NASDAQ, VIX,
rates) shows no significant *lagged* daily effect, which is unsurprising —
same-day market beta is a different question than whether yesterday's level
predicts today's return.

This is deliberately kept separate from the 12-month Monte Carlo forecast:
an exogenous-regressor forecast needs *future* values of every regressor for
the full horizon, and nobody has a credible 12-month path for the NASDAQ or
VIX. `simulate_price_paths`'s `drift_overrides` hook is where a bounded,
short-horizon version of this (drivers held at their last known level) would
plug into the forward simulation.

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
