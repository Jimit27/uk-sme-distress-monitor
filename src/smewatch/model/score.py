"""Score the newest filings with the trained insolvency model."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from smewatch import config
from smewatch.model.explain import reason_codes, template_summary

log = logging.getLogger(__name__)

DISPLAY_COLUMNS = [
    "company_number", "entity_name", "filing_batch", "balance_sheet_date", "sic_section", "sic_section_name",
    "postcode_area", "post_town", "current_status", "total_assets", "equity", "equity_prior", "cash",
    "current_assets", "current_liabilities", "employees", "approx_age_years", "filed_late",
]


def score_frame(df: pd.DataFrame, model_dir: Path | None = None) -> pd.DataFrame:
    model_dir = model_dir or config.MODELS
    ins = joblib.load(model_dir / "insolvency_model.joblib")
    fail = joblib.load(model_dir / "failure_model.joblib")

    out = df[[c for c in DISPLAY_COLUMNS if c in df.columns]].copy()
    out["pd"] = ins.predict_pd(df)
    out["grade"] = ins.grade(out["pd"].to_numpy())
    out["pd_failure"] = fail.predict_pd(df)
    reasons = reason_codes(ins.contributions(df))
    out["reason_1"] = [r[0][0] if len(r) > 0 else None for r in reasons]
    out["reason_2"] = [r[1][0] if len(r) > 1 else None for r in reasons]
    out["reason_3"] = [r[2][0] if len(r) > 2 else None for r in reasons]
    out["summary"] = [template_summary(row, r) for (_, row), r in zip(out.iterrows(), reasons)]
    return out.sort_values("pd", ascending=False).reset_index(drop=True)


def score_latest(warehouse: Path | None = None, out_path: Path | None = None) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(str(warehouse or config.WAREHOUSE))
    try:
        df = con.execute("select * from fct_live_filings").df()
    finally:
        con.close()
    if df.empty:
        log.warning("no live filings to score")
        return df
    scored = score_frame(df)
    out_path = out_path or config.PUBLISHED / "live_scores.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out_path, index=False)
    log.info("scored %d live filings -> %s", len(scored), out_path)
    return scored
