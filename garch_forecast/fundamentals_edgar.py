"""Fabrinet's actual reported financials, pulled from SEC EDGAR's free XBRL API --
no API key needed. This is the "financial performance" driver (revenue growth,
margins, EPS, buybacks) mentioned as a price driver for FN: real quarterly figures
straight from 10-Q/10-K filings, not a hand-typed snapshot.

IBKR's fundamentals endpoint (see ibkr_data.py) needs a Reuters Fundamentals data
subscription this account doesn't have (Error 10358); EDGAR is a solid substitute
since it's the primary source those vendor feeds repackage anyway.
"""

from __future__ import annotations

import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

FABRINET_CIK = "0001408710"
EDGAR_BASE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
# SEC requires a descriptive User-Agent identifying the requester; see
# https://www.sec.gov/os/webmaster-faq#developers
USER_AGENT = "uchi-ai-investing-comp research (contact: set via SEC_EDGAR_CONTACT env var)"

# us-gaap XBRL tag -> output column name
FACT_TAGS = {
    "RevenueFromContractWithCustomerIncludingAssessedTax": "revenue",
    "NetIncomeLoss": "net_income",
    "EarningsPerShareDiluted": "eps_diluted",
    "GrossProfit": "gross_profit",
    "CostOfRevenue": "cost_of_revenue",
    "PaymentsForRepurchaseOfCommonStock": "buybacks",
}

MIN_QUARTER_DAYS = 80
MAX_QUARTER_DAYS = 100


def fetch_company_facts(cik: str = FABRINET_CIK, contact: str | None = None) -> dict:
    ua = USER_AGENT if not contact else f"uchi-ai-investing-comp research (contact: {contact})"
    resp = requests.get(EDGAR_BASE.format(cik=cik), headers={"User-Agent": ua}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _single_quarter_frame(facts: dict, tag: str) -> pd.DataFrame:
    """XBRL reports cumulative fiscal-year-to-date figures alongside discrete-quarter
    ones under the same tag; keeps only entries spanning ~one quarter (80-100 days).
    Fabrinet's fiscal Q4 is never tagged as a standalone 3-month period (only the
    full-year 10-K figure, which this filters out), so the resulting series has a
    gap every 4th quarter -- callers must match by `period_end` date, not row
    position, when computing YoY changes.

    Indexed by period_end (unique, gap-tolerant); carries `filed` as a column so
    callers can know when a value actually became public.
    """
    if tag not in facts.get("facts", {}).get("us-gaap", {}):
        return pd.DataFrame(columns=["filed", "val"])

    unit_key = next(iter(facts["facts"]["us-gaap"][tag]["units"]))
    rows = facts["facts"]["us-gaap"][tag]["units"][unit_key]

    quarterly = []
    for r in rows:
        if "start" not in r or "end" not in r:
            continue
        start, end = pd.Timestamp(r["start"]), pd.Timestamp(r["end"])
        days = (end - start).days
        if MIN_QUARTER_DAYS <= days <= MAX_QUARTER_DAYS:
            quarterly.append({"period_end": end, "filed": pd.Timestamp(r["filed"]), "val": r["val"]})

    if not quarterly:
        return pd.DataFrame(columns=["filed", "val"])

    # Keep each period_end's *first* disclosure (not later restatements) -- what
    # matters here is when the market first learned the number, not corrections.
    df = pd.DataFrame(quarterly).sort_values("filed").drop_duplicates("period_end", keep="first")
    return df.set_index("period_end").sort_index()[["filed", "val"]]


def build_quarterly_fundamentals(cik: str = FABRINET_CIK, contact: str | None = None) -> pd.DataFrame:
    """Returns a DataFrame indexed by fiscal period end, one row per (non-Q4) quarter,
    with derived margin/growth columns and a `filed` date per row for downstream
    look-ahead-safe alignment onto a daily panel.
    """
    facts = fetch_company_facts(cik, contact=contact)

    frames = {}
    for tag, col in FACT_TAGS.items():
        f = _single_quarter_frame(facts, tag)
        if not f.empty:
            frames[col] = f
        else:
            logger.warning("No single-quarter data found for tag %s", tag)

    values = pd.concat({col: f["val"] for col, f in frames.items()}, axis=1).sort_index()
    # revenue/net_income/EPS are always reported together, so their filing date is
    # representative of when the whole quarter's figures became public.
    filed = frames["revenue"]["filed"].reindex(values.index)
    for col in ("net_income", "eps_diluted"):
        filed = filed.fillna(frames[col]["filed"].reindex(values.index))
    values["filed"] = filed

    # RevenueFromContractWithCustomerIncludingAssessedTax only exists from ASC 606
    # adoption (~2018) onward; older rows have no revenue tagged under it at all
    # (and are old enough not to matter for the vol model anyway).
    values = values.dropna(subset=["revenue"])

    values["gross_margin"] = values["gross_profit"] / values["revenue"]
    values["net_margin"] = values["net_income"] / values["revenue"]

    # YoY growth matched by actual calendar distance (350-380 days back), not row
    # position -- the missing Q4 makes a plain .pct_change(4) silently wrong.
    prior_idx = values.index.map(lambda d: _nearest_prior_period(values.index, d))
    values["revenue_growth_yoy"] = values["revenue"].values / values["revenue"].reindex(prior_idx).values - 1
    values["eps_growth_yoy"] = values["eps_diluted"].values / values["eps_diluted"].reindex(prior_idx).values - 1
    return values


def _nearest_prior_period(index: pd.DatetimeIndex, date: pd.Timestamp, lo: int = 350, hi: int = 380) -> pd.Timestamp | None:
    candidates = [d for d in index if lo <= (date - d).days <= hi]
    return max(candidates) if candidates else None
