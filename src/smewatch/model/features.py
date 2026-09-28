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
    "retained_earnings_to_assets",
    "bank_borrowings_to_assets",
    "director_loans_to_assets",
    "tax_payable_to_assets",
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
    "tax_payable_to_assets": 1,
}

# (description when the feature pushes risk UP, description when it pushes risk DOWN)
DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "log_total_assets": ("small balance sheet", "larger balance sheet"),
    "log_employees": ("few or no employees reported", "larger workforce"),
    "n_officers": ("officer structure typical of higher-risk filers", "officer structure typical of lower-risk filers"),
    "log_n_facts": ("very sparse accounts filing", "detailed accounts filing"),
    "equity_to_assets": ("thin or negative equity cushion", "strong equity cushion"),
    "negative_equity": ("liabilities exceed assets (negative equity)", "positive net assets"),
    "liabilities_to_assets": ("high leverage", "low leverage"),
    "long_term_creditors_to_assets": ("heavy long-term debt", "little long-term debt"),
    "retained_earnings_to_assets": ("accumulated losses", "accumulated profits"),
    "bank_borrowings_to_assets": ("reliance on bank borrowing", "little bank borrowing"),
    "director_loans_to_assets": ("propped up by director loans", "little reliance on director loans"),
    "tax_payable_to_assets": ("large tax and social security liabilities", "small tax liabilities"),
    "current_ratio": ("weak short-term liquidity", "comfortable short-term liquidity"),
    "working_capital_to_assets": ("negative working capital", "positive working capital"),
    "cash_to_assets": ("low cash relative to assets", "healthy cash position"),
    "cash_to_current_liabilities": ("cash covers little of short-term debts", "cash covers short-term debts"),
    "debtors_to_assets": ("assets tied up in debtors", "few assets tied up in debtors"),
    "current_liabilities_to_assets": ("heavy short-term liabilities", "light short-term liabilities"),
    "equity_change_to_assets": ("equity fell over the year", "equity grew over the year"),
    "asset_growth": ("balance sheet movement typical of distress", "stable or growing balance sheet"),
    "cash_change_to_assets": ("cash fell over the year", "cash grew over the year"),
    "current_liabilities_change_to_assets": ("short-term liabilities rising", "short-term liabilities falling"),
    "equity_turned_negative": ("equity turned negative this year", "equity did not turn negative"),
    "has_prior_year": ("no prior-year comparatives (young or first filing)", "established filing history"),
    "discloses_turnover": ("filing pattern of higher-risk companies", "filing pattern of lower-risk companies"),
    "filing_lag_days": ("accounts filed late in the window", "accounts filed promptly"),
    "filed_late": ("accounts filed after the statutory deadline", "accounts filed on time"),
    "approx_age_years": ("young company", "long-established company"),
    "entity_england_wales": ("registration type", "registration type"),
    "entity_scotland": ("registration type", "registration type"),
    "entity_northern_ireland": ("registration type", "registration type"),
    "entity_llp": ("registration type", "registration type"),
    "entity_other": ("registration type", "registration type"),
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
