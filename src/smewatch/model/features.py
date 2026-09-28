"""Model feature set and human-readable descriptions used for reason codes."""

from __future__ import annotations

# Every feature is derived from the filed accounts alone (see
# dbt/models/intermediate/int_accounts__features.sql).
NUMERIC_FEATURES: list[str] = [
    "log_total_assets",
    "log_employees",
    "n_officers",
    "log_n_facts",
    "equity_to_assets",
    "negative_equity",
    "liabilities_to_assets",
    "long_term_creditors_to_assets",
    "current_ratio",
    "working_capital_to_assets",
    "cash_to_assets",
    "cash_to_current_liabilities",
    "debtors_to_assets",
    "current_liabilities_to_assets",
    "equity_change_to_assets",
    "asset_growth",
    "cash_change_to_assets",
    "current_liabilities_change_to_assets",
    "equity_turned_negative",
    "has_prior_year",
    "discloses_turnover",
    "filing_lag_days",
    "filed_late",
    "approx_age_years",
]

BINARY_FEATURES: set[str] = {
    "negative_equity", "equity_turned_negative", "has_prior_year", "discloses_turnover", "filed_late",
    "entity_england_wales", "entity_scotland", "entity_northern_ireland", "entity_llp", "entity_other",
}

CATEGORICAL_FEATURES: list[str] = ["entity_group"]
ENTITY_GROUPS = ["england_wales", "scotland", "northern_ireland", "llp", "other"]

FEATURES = NUMERIC_FEATURES + [f"entity_{g}" for g in ENTITY_GROUPS]

# Monotone constraints for the gradient-boosted model: +1 means risk may only
# rise with the feature, -1 only fall, 0 unconstrained. Constraining the
# obvious credit relationships stops the model learning noise the wrong way
# round, and makes its explanations defensible to a credit committee.
MONOTONE: dict[str, int] = {
    "equity_to_assets": -1,
    "negative_equity": 1,
    "liabilities_to_assets": 1,
    "cash_to_assets": -1,
    "cash_to_current_liabilities": -1,
    "working_capital_to_assets": -1,
    "equity_turned_negative": 1,
    "filed_late": 1,
}

# Plain-English phrases for each feature: (when the value is HIGH, when it is LOW),
# judged against the training median. Reason codes pick the phrase that matches
# the company's actual value, so the text is right whichever way the model
# reads the feature (e.g. size raises formal-insolvency risk but lowers
# strike-off risk).
DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "log_total_assets": ("larger balance sheet, so more creditors with a claim", "very small balance sheet"),
    "log_employees": ("larger workforce", "few or no employees"),
    "n_officers": ("more officers than typical", "fewer officers than typical"),
    "log_n_facts": ("detailed accounts filing", "very sparse accounts filing"),
    "equity_to_assets": ("strong equity cushion", "thin or negative equity cushion"),
    "negative_equity": ("liabilities exceed assets (negative equity)", "positive net assets"),
    "liabilities_to_assets": ("high leverage", "low leverage"),
    "long_term_creditors_to_assets": ("heavy long-term debt", "little long-term debt"),
    "current_ratio": ("comfortable short-term liquidity", "weak short-term liquidity"),
    "working_capital_to_assets": ("positive working capital", "negative or thin working capital"),
    "cash_to_assets": ("healthy cash position", "low cash relative to assets"),
    "cash_to_current_liabilities": ("cash covers short-term debts", "cash covers little of short-term debts"),
    "debtors_to_assets": ("assets tied up in debtors", "few trade debtors"),
    "current_liabilities_to_assets": ("heavy short-term liabilities", "light short-term liabilities"),
    "equity_change_to_assets": ("equity grew over the year", "equity fell over the year"),
    "asset_growth": ("balance sheet grew over the year", "balance sheet shrank over the year"),
    "cash_change_to_assets": ("cash rose over the year", "cash fell over the year"),
    "current_liabilities_change_to_assets": ("short-term liabilities rising", "short-term liabilities falling"),
    "equity_turned_negative": ("equity turned negative this year", "equity did not turn negative"),
    "has_prior_year": ("established filing history", "no prior-year comparatives (young or first filing)"),
    "discloses_turnover": ("turnover disclosed", "no turnover disclosed"),
    "filing_lag_days": ("accounts filed late in the filing window", "accounts filed promptly"),
    "filed_late": ("accounts filed after the statutory deadline", "accounts filed on time"),
    "approx_age_years": ("long-established company", "young company"),
    "entity_england_wales": ("registered in England & Wales", "registered outside England & Wales"),
    "entity_scotland": ("registered in Scotland", "not registered in Scotland"),
    "entity_northern_ireland": ("registered in Northern Ireland", "not registered in Northern Ireland"),
    "entity_llp": ("limited liability partnership", "not an LLP"),
    "entity_other": ("unusual registration type", "standard registration type"),
}


def add_entity_dummies(df):
    """One-hot encode entity_group into fixed columns (stable across train and scoring)."""
    out = df.copy()
    groups = out.get("entity_group")
    for g in ENTITY_GROUPS:
        out[f"entity_{g}"] = (groups == g).astype(float) if groups is not None else 0.0
    return out


def model_matrix(df):
    """Return the feature matrix in canonical column order, as float."""
    x = add_entity_dummies(df)
    for c in FEATURES:
        if c not in x:
            x[c] = float("nan")
    return x[FEATURES].astype(float)
