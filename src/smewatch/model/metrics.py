"""Evaluation metrics for rare-event risk models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def capture_at(y: np.ndarray, p: np.ndarray, share: float) -> float:
    """Share of all events found in the riskiest ``share`` of companies."""
    n = max(1, int(round(len(p) * share)))
    idx = np.argsort(-p, kind="stable")[:n]
    total = y.sum()
    return float(y[idx].sum() / total) if total else float("nan")


def lift_at(y: np.ndarray, p: np.ndarray, share: float) -> float:
    base = y.mean()
    return capture_at(y, p, share) / share if base else float("nan")


def bootstrap_auc(y: np.ndarray, p: np.ndarray, reps: int = 200, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        stats.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def summarise(y, p, bootstrap: bool = True, probabilities: bool = True) -> dict:
    """Ranking and calibration metrics. Set probabilities=False for raw scores."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    out = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "gini": float(2 * roc_auc_score(y, p) - 1),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)) if probabilities else None,
        "mean_predicted": float(p.mean()) if probabilities else None,
        "capture_top_1pct": capture_at(y, p, 0.01),
        "capture_top_5pct": capture_at(y, p, 0.05),
        "capture_top_10pct": capture_at(y, p, 0.10),
        "lift_top_1pct": lift_at(y, p, 0.01),
        "lift_top_10pct": lift_at(y, p, 0.10),
    }
    if bootstrap:
        lo, hi = bootstrap_auc(y, p)
        out["roc_auc_ci95"] = [lo, hi]
    return out


def decile_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted event rate by risk decile (decile 10 = riskiest)."""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "p": np.asarray(p).astype(float)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=range(1, bins + 1)).astype(int)
    t = df.groupby("decile").agg(companies=("y", "size"), events=("y", "sum"), observed_rate=("y", "mean"), predicted_rate=("p", "mean"))
    t["share_of_events"] = t["events"] / max(1, t["events"].sum())
    return t.reset_index()
