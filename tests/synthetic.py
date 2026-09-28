"""Synthetic cohort generator for unit tests (shape-compatible with fct_training_cohort)."""
import numpy as np
import pandas as pd

from smewatch.model.features import NUMERIC_FEATURES


def make_cohort(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f: rng.normal(size=n) for f in NUMERIC_FEATURES})
    df["negative_equity"] = (df["equity_to_assets"] < -0.8).astype(int)
    df["filed_late"] = (rng.random(n) < 0.1).astype(int)
    # risk driven by leverage, cash and late filing
    logit = -4.0 - 1.2 * df["equity_to_assets"] - 0.8 * df["cash_to_assets"] + 1.0 * df["filed_late"]
    p = 1 / (1 + np.exp(-logit))
    df["label_insolvency"] = (rng.random(n) < p).astype(int)
    df["label_failure"] = np.maximum(df["label_insolvency"], (rng.random(n) < 0.08).astype(int))
    df.loc[rng.random(n) < 0.1, "cash_to_assets"] = np.nan  # missing values
    df["entity_group"] = rng.choice(["england_wales", "scotland", "llp"], n, p=[0.85, 0.1, 0.05])
    df["company_number"] = [f"{i:08d}" for i in range(n)]
    df["filing_batch"] = np.where(np.arange(n) < n * 0.5, "July2025", "August2025")
    df["split"] = np.where(df["filing_batch"] == "July2025", "train", "test")
    df["entity_name"] = "SYNTHETIC LTD"
    df["outcome_group"] = np.where(df["label_insolvency"] == 1, "insolvency", "active")
    df["horizon_days"] = np.where(df["split"] == "train", 412, 381)
    return df
