"""Reason codes and plain-English risk summaries.

Each score comes with the three features that pushed it furthest towards
distress (largest positive SHAP contributions), translated into words. The
summary is deterministic by default; if ANTHROPIC_API_KEY is set, an optional
LLM rewrite turns the same facts into a short analyst-style note. The LLM
only rephrases the numbers it is given and is never asked to invent them.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from smewatch.model.features import ADVERSE_SIDE, BINARY_FEATURES, DESCRIPTIONS

GRADE_WORDS = {
    "A": "low",
    "B": "below-average",
    "C": "elevated",
    "D": "high",
    "E": "very high",
}


def reason_codes(contrib: pd.DataFrame, k: int = 3) -> list[list[tuple[str, float]]]:
    """Top-k (feature, SHAP contribution) pairs that increase risk, per row."""
    out = []
    values = contrib.to_numpy()
    cols = np.array(contrib.columns)
    for row in values:
        order = np.argsort(-row)
        top = [(cols[i], float(row[i])) for i in order[:k] if row[i] > 0]
        out.append(top)
    return out


def _is_high(feature: str, value: float, median: float | None) -> bool:
    if feature in BINARY_FEATURES or median is None or pd.isna(median):
        median = 0.5
    return value > median


def select_reasons(
    contrib: pd.DataFrame,
    values: pd.DataFrame,
    medians: dict,
    k: int = 3,
) -> list[list[tuple[str, float]]]:
    """Top-k risk-increasing features per row, skipping ones stated on their benign side."""
    out = []
    cols = np.array(contrib.columns)
    for c_row, (_, v_row) in zip(contrib.to_numpy(), values.iterrows()):
        picked: list[tuple[str, float]] = []
        for i in np.argsort(-c_row):
            if c_row[i] <= 0 or len(picked) == k:
                break
            f = cols[i]
            side = ADVERSE_SIDE.get(f)
            v = v_row.get(f)
            if side and v is not None and not pd.isna(v):
                if (side == "high") != _is_high(f, v, medians.get(f)):
                    continue
            picked.append((f, float(c_row[i])))
        out.append(picked)
    return out


def describe(feature: str, value: float | None = None, median: float | None = None) -> str:
    """Phrase for a feature given the company's value relative to the typical filer."""
    high, low = DESCRIPTIONS.get(feature, (feature.replace("_", " "), feature.replace("_", " ")))
    if value is None or pd.isna(value):
        return f"{feature.replace('_', ' ')} not reported"
    return high if _is_high(feature, value, median) else low


def fmt_money(v: float | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e6:
        return f"{sign}£{v / 1e6:.1f}m"
    if v >= 1e3:
        return f"{sign}£{v / 1e3:.0f}k"
    return f"{sign}£{v:.0f}"


def template_summary(
    row: pd.Series,
    reasons: list[tuple[str, float]],
    values: pd.Series | None = None,
    medians: dict | None = None,
) -> str:
    grade = row.get("grade", "?")
    pd_pct = 100 * float(row.get("pd", float("nan")))
    name = (row.get("entity_name") or row.get("company_number") or "This company").strip()
    parts = [
        f"{name} is graded {grade} ({GRADE_WORDS.get(grade, 'unknown')} risk), "
        f"with an estimated {pd_pct:.1f}% chance of entering insolvency within 12 months."
    ]
    facts = []
    if pd.notna(row.get("total_assets")):
        facts.append(f"total assets of {fmt_money(row['total_assets'])}")
    if pd.notna(row.get("equity")):
        facts.append(f"net assets of {fmt_money(row['equity'])}")
    if pd.notna(row.get("cash")):
        facts.append(f"cash of {fmt_money(row['cash'])}")
    if facts:
        bsd = row.get("balance_sheet_date")
        when = f" at {pd.Timestamp(bsd):%d %b %Y}" if pd.notna(bsd) else ""
        parts.append(f"Its latest accounts show {', '.join(facts)}{when}.")
    if reasons:
        phrases = [
            describe(f, None if values is None else values.get(f), (medians or {}).get(f)) for f, _ in reasons
        ]
        parts.append("Main risk drivers: " + "; ".join(phrases) + ".")
    else:
        parts.append("No individual factor pushes its risk above the typical filer.")
    return " ".join(parts)


def llm_summary(
    row: pd.Series,
    reasons: list[tuple[str, float]],
    values: pd.Series | None = None,
    medians: dict | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> str | None:
    """Optional: rewrite the template facts as a two-sentence analyst note."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        return None
    facts = template_summary(row, reasons, values, medians)
    prompt = (
        "You are a UK credit analyst. Rewrite the following facts as a concise two-sentence note "
        "for a lending team. Use only the facts given; do not add numbers or speculation.\n\n" + facts
    )
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(model=model, max_tokens=200, messages=[{"role": "user", "content": prompt}])
    return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text").strip() or None
