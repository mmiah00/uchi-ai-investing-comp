"""Quarterly gross-margin trends for FN and AI-optics-adjacent peers, pulled from
SEC EDGAR (same approach as fundamentals_edgar.py, generalized across companies).

Built to answer one question: is FN's flat/declining gross margin during the AI
buildout an industry-wide pattern (contract manufacturers ramping capacity ahead
of utilization) or something specific to FN? See notebooks/fn_short_thesis_review.ipynb.
"""

from __future__ import annotations

import pandas as pd

from .fundamentals_edgar import _single_quarter_frame, fetch_company_facts

PEER_CIKS = {
    "FN": "0001408710",  # Fabrinet
    "COHR": "0000820318",  # Coherent Corp (optical components/lasers, AI-datacenter exposed)
    "LITE": "0001633978",  # Lumentum Holdings (optical components, AI-datacenter exposed)
    "CLS": "0001030894",  # Celestica (EMS/contract manufacturer, closest business-model comp)
    "JBL": "0000898293",  # Jabil (large diversified EMS, lower-margin baseline)
}

# Tried in priority order per company since XBRL tag usage varies.
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]
COST_TAGS = ["CostOfGoodsAndServicesSold", "CostOfRevenue"]
GROSS_PROFIT_TAGS = ["GrossProfit"]


def _first_available(facts: dict, tags: list[str]) -> tuple[str | None, pd.DataFrame]:
    available = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        if tag in available:
            frame = _single_quarter_frame(facts, tag)
            if not frame.empty:
                return tag, frame
    return None, pd.DataFrame(columns=["filed", "val"])


def quarterly_gross_margin(cik: str, contact: str | None = None) -> tuple[pd.DataFrame, str, str]:
    """Returns (DataFrame[revenue, gross_profit, gross_margin] indexed by period_end,
    revenue_tag_used, gross_profit_source) for one company.
    """
    facts = fetch_company_facts(cik, contact=contact)
    rev_tag, rev = _first_available(facts, REVENUE_TAGS)
    gp_tag, gp = _first_available(facts, GROSS_PROFIT_TAGS)

    if gp.empty:
        # Not every filer tags GrossProfit directly (e.g. Coherent) -- derive it.
        cost_tag, cost = _first_available(facts, COST_TAGS)
        joined = rev[["val"]].join(cost[["val"]], lsuffix="_rev", rsuffix="_cost", how="inner")
        gross_profit = joined["val_rev"] - joined["val_cost"]
        out = pd.DataFrame({"revenue": joined["val_rev"], "gross_profit": gross_profit})
        gp_source = f"derived (revenue - {cost_tag})"
    else:
        joined = rev[["val"]].join(gp[["val"]], lsuffix="_rev", rsuffix="_gp", how="inner")
        out = pd.DataFrame({"revenue": joined["val_rev"], "gross_profit": joined["val_gp"]})
        gp_source = "GrossProfit tag"

    out["gross_margin"] = out["gross_profit"] / out["revenue"]
    return out.sort_index(), rev_tag, gp_source


def build_peer_panel(contact: str | None = None, tickers: dict | None = None) -> dict[str, pd.DataFrame]:
    tickers = tickers or PEER_CIKS
    return {name: quarterly_gross_margin(cik, contact=contact)[0] for name, cik in tickers.items()}
