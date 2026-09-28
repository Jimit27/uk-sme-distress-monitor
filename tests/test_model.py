import numpy as np
import pandas as pd
import pytest

from smewatch.model.explain import reason_codes, template_summary
from smewatch.model.features import FEATURES, model_matrix
from smewatch.model.metrics import capture_at, decile_table, summarise
from smewatch.model.score import score_frame
from smewatch.model.train import train_target
from synthetic import make_cohort


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    df = make_cohort()
    res = {t: train_target(df, t, out) for t in ("insolvency", "failure")}
    return df, res, out


def test_model_matrix_is_stable():
    df = pd.DataFrame({"equity_to_assets": [0.1], "entity_group": ["scotland"]})
    x = model_matrix(df)
    assert list(x.columns) == FEATURES
    assert x.loc[0, "entity_scotland"] == 1.0 and np.isnan(x.loc[0, "cash_to_assets"])


def test_models_beat_naive_rule(trained):
    _, res, _ = trained
    t = res["insolvency"]["test"]
    assert t["lightgbm_calibrated"]["roc_auc"] > t["rule_negative_equity"]["roc_auc"]
    assert t["logistic_regression"]["roc_auc"] > 0.75


def test_calibration_is_reasonable(trained):
    _, res, _ = trained
    m = res["insolvency"]["test"]["lightgbm_calibrated"]
    assert abs(m["mean_predicted"] - m["base_rate"]) < 0.02


def test_grades_rank_risk(trained):
    _, res, _ = trained
    rates = {g["grade"]: g["observed_rate"] for g in res["insolvency"]["grade_table"]}
    assert rates["E"] > rates["A"]


def test_score_frame_and_reasons(trained):
    df, _, out = trained
    live = df[df["split"] == "test"].head(50).copy()
    live["total_assets"] = 1000.0
    scored = score_frame(live, model_dir=out)
    assert len(scored) == 50
    assert scored["pd"].between(0, 1).all()
    assert scored["pd"].is_monotonic_decreasing
    assert set(scored["grade"]) <= set("ABCDE")
    assert scored["summary"].str.contains("chance of entering insolvency").all()


def test_reason_codes_only_positive():
    contrib = pd.DataFrame({"a": [0.5, -0.1], "b": [0.2, -0.3], "c": [-1.0, 0.05]})
    rc = reason_codes(contrib, k=3)
    assert rc[0] == [("a", 0.5), ("b", 0.2)]
    assert rc[1] == [("c", 0.05)]


def test_template_summary_handles_missing():
    row = pd.Series({"grade": "D", "pd": 0.12, "entity_name": "ACME LTD", "total_assets": np.nan})
    s = template_summary(row, [("negative_equity", 0.4)])
    assert "ACME LTD is graded D" in s and "negative equity" in s


def test_metrics_helpers():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.9, 0.8])
    assert capture_at(y, p, 0.4) == 1.0
    s = summarise(y, p, bootstrap=False)
    assert s["roc_auc"] == 1.0 and s["ks"] == 1.0
    assert len(decile_table(np.tile(y, 4), np.tile(p, 4) + np.arange(20) * 1e-6, bins=5)) == 5
