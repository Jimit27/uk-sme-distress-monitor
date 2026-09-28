"""Train and evaluate the distress models.

Design
------
* Training rows: accounts filed in the training month(s).
* Out-of-time test: accounts filed in the *following* month, never seen in
  training (different companies, later filing window).
* Two targets: formal insolvency within ~12 months (headline) and the broader
  failure/exit outcome.
* Two models per target:
    - logistic regression on imputed, standardised features: the transparent
      benchmark a credit team would start from;
    - LightGBM with monotone constraints on the textbook credit relationships,
      calibrated with isotonic regression on a held-out slice of the training
      month.
* A naive single-rule benchmark (negative equity) keeps the headline numbers
  honest: the models have to beat what an analyst could do by eye.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from smewatch import config
from smewatch.model.features import FEATURES, MONOTONE, model_matrix
from smewatch.model.metrics import decile_table, summarise

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*eval_set.*", category=Warning)

TARGETS = {
    "insolvency": "label_insolvency",
    "failure": "label_failure",
}

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    n_estimators=3000,
    verbose=-1,
)


@dataclass
class DistressModel:
    """A fitted, calibrated model plus everything needed to score new filings."""

    target: str
    booster: lgb.LGBMClassifier
    calibrator: IsotonicRegression
    features: list[str]
    grade_cutoffs: list[float]
    metadata: dict

    def raw_score(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(model_matrix(df))[:, 1]

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.raw_score(df))

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-company SHAP values (log-odds contributions) from LightGBM."""
        contrib = self.booster.predict_proba(model_matrix(df), pred_contrib=True)
        return pd.DataFrame(contrib[:, :-1], columns=self.features, index=df.index)

    def grade(self, pd_values: np.ndarray) -> np.ndarray:
        """Map PDs to letter grades A (safest) .. E (riskiest)."""
        idx = np.searchsorted(self.grade_cutoffs, pd_values, side="right")
        return np.array(list("ABCDE"))[idx]


def load_cohort(warehouse: Path | None = None) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(str(warehouse or config.WAREHOUSE))
    try:
        return con.execute("select * from fct_training_cohort").df()
    finally:
        con.close()


def _monotone_vector() -> list[int]:
    return [MONOTONE.get(f, 0) for f in FEATURES]


def fit_logistic(x: pd.DataFrame, y: np.ndarray):
    pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000),
    )
    pipe.fit(x, y)
    return pipe


def fit_lgbm(x_tr, y_tr, x_val, y_val, seed: int = 42) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**LGB_PARAMS, monotone_constraints=_monotone_vector(), random_state=seed)
    model.fit(
        x_tr,
        y_tr,
        eval_set=[(x_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return model


def grade_cutoffs(pd_values: np.ndarray) -> list[float]:
    """Grade boundaries at the 50th/75th/90th/97th percentiles of predicted PD.

    A: safest half of filers, B: next quarter, C: next 15%, D: next 7%, E: riskiest 3%.
    """
    return [float(np.quantile(pd_values, q)) for q in (0.50, 0.75, 0.90, 0.97)]


def train_target(df: pd.DataFrame, target: str, out_dir: Path, seed: int = 42) -> dict:
    label = TARGETS[target]
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    x_train_all, y_train_all = model_matrix(train), train[label].to_numpy()
    fit_idx, cal_idx = train_test_split(np.arange(len(train)), test_size=0.25, stratify=y_train_all, random_state=seed)
    x_fit, y_fit = x_train_all.iloc[fit_idx], y_train_all[fit_idx]
    x_cal, y_cal = x_train_all.iloc[cal_idx], y_train_all[cal_idx]
    x_test, y_test = model_matrix(test), test[label].to_numpy()

    log.info("[%s] train=%d (events %d)  test=%d (events %d)", target, len(train), y_train_all.sum(), len(test), y_test.sum())

    # 1. benchmark rule
    rule = test["negative_equity"].fillna(0).to_numpy() + 1e-3 * test["liabilities_to_assets"].fillna(0).to_numpy()
    # 2. logistic regression
    logit = fit_logistic(x_fit, y_fit)
    # 3. monotone LightGBM + isotonic calibration on the held-out training slice
    booster = fit_lgbm(x_fit, y_fit, x_cal, y_cal, seed=seed)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(booster.predict_proba(x_cal)[:, 1], y_cal)

    p_logit = logit.predict_proba(x_test)[:, 1]
    p_raw = booster.predict_proba(x_test)[:, 1]
    p_cal = iso.predict(p_raw)
    # isotonic produces ties; rank on the raw score to break them for ranking metrics
    p_rank = p_cal * (1 - 1e-9) + 1e-9 * p_raw

    results = {
        "target": target,
        "label": label,
        "train_batches": sorted(train["filing_batch"].unique().tolist()),
        "test_batches": sorted(test["filing_batch"].unique().tolist()),
        "best_iteration": int(booster.best_iteration_ or booster.n_estimators),
        "test": {
            "rule_negative_equity": summarise(y_test, rule, bootstrap=False, probabilities=False),
            "logistic_regression": summarise(y_test, p_logit),
            "lightgbm_calibrated": summarise(y_test, p_rank),
        },
        "train_in_sample_auc": float(
            summarise(y_fit, booster.predict_proba(x_fit)[:, 1], bootstrap=False)["roc_auc"]
        ),
    }
    cut = grade_cutoffs(iso.predict(booster.predict_proba(x_train_all)[:, 1]))
    model = DistressModel(
        target=target,
        booster=booster,
        calibrator=iso,
        features=list(FEATURES),
        grade_cutoffs=cut,
        metadata={k: v for k, v in results.items() if k != "test"} | {"test_auc": results["test"]["lightgbm_calibrated"]["roc_auc"]},
    )

    grades = model.grade(p_cal)
    grade_tbl = (
        pd.DataFrame({"grade": grades, "y": y_test, "pd": p_cal})
        .groupby("grade")
        .agg(companies=("y", "size"), events=("y", "sum"), observed_rate=("y", "mean"), mean_pd=("pd", "mean"))
        .reset_index()
    )
    results["grade_table"] = grade_tbl.to_dict(orient="records")
    results["grade_cutoffs"] = cut
    results["decile_table"] = decile_table(y_test, p_rank).to_dict(orient="records")

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "gain": booster.booster_.feature_importance("gain"),
            "mean_abs_shap": np.abs(model.contributions(test)).mean().reindex(FEATURES).to_numpy(),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    results["feature_importance"] = importance.to_dict(orient="records")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f"{target}_model.joblib")
    joblib.dump(logit, out_dir / f"{target}_logistic.joblib")

    preds = test[["company_number", "filing_batch", "entity_name", label, "outcome_group"]].copy()
    preds["pd"] = p_cal
    preds["pd_logistic"] = p_logit
    preds["grade"] = grades
    results["_predictions"] = preds
    return results


def train_all(warehouse: Path | None = None, model_dir: Path | None = None, report_dir: Path | None = None) -> dict:
    df = load_cohort(warehouse)
    model_dir = model_dir or config.MODELS
    report_dir = report_dir or config.REPORTS
    report_dir.mkdir(parents=True, exist_ok=True)

    cohort = {
        "rows": int(len(df)),
        "by_split": df.groupby("split").size().to_dict(),
        "outcome_mix": df.groupby(["split", "outcome_group"]).size().rename("n").reset_index().to_dict(orient="records"),
        "median_horizon_days": df.groupby("split")["horizon_days"].median().astype(float).to_dict(),
    }
    all_results = {"cohort": cohort, "models": {}}
    for target in TARGETS:
        res = train_target(df, target, model_dir)
        preds = res.pop("_predictions")
        preds.to_parquet(report_dir / f"test_predictions_{target}.parquet", index=False)
        all_results["models"][target] = res
        m = res["test"]["lightgbm_calibrated"]
        log.info("[%s] test AUC %.3f  PR-AUC %.3f  top-10%% capture %.1f%%", target, m["roc_auc"], m["pr_auc"], 100 * m["capture_top_10pct"])

    (report_dir / "metrics.json").write_text(json.dumps(all_results, indent=2, default=float))
    return all_results
