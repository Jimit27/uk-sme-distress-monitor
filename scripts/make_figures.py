"""Render the README figures from reports/metrics.json and the published marts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
PUB = ROOT / "data" / "published"

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
GRADES = {"A": "#86b6ef", "B": "#5598e7", "C": "#2a78d6", "D": "#1c5cab", "E": "#104281"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
    }
)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


def capture_curve(preds: pd.DataFrame, label: str):
    y = preds[label].to_numpy()
    share = np.arange(1, len(y) + 1) / len(y) * 100
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for col, name, color in [("pd", "LightGBM (calibrated)", SERIES[0]), ("pd_logistic", "Logistic regression", SERIES[1])]:
        order = np.argsort(-preds[col].to_numpy(), kind="stable")
        ax.plot(share, np.cumsum(y[order]) / y.sum() * 100, color=color, lw=2, label=name)
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls=":", label="Random")
    order = np.argsort(-preds["pd"].to_numpy(), kind="stable")
    at10 = np.cumsum(y[order])[int(len(y) * 0.1) - 1] / y.sum() * 100
    ax.scatter([10], [at10], s=36, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"{at10:.0f}% of insolvencies\nin the riskiest 10%", (10, at10), xytext=(24, at10 - 22), color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Companies reviewed, riskiest first (%)", ylabel="Insolvencies caught (%)")
    ax.set_title("Insolvencies caught vs companies reviewed (test month)")
    ax.legend(loc="lower right")
    save(fig, "capture_curve.png")


def grade_rates(grade_table: list[dict]):
    g = pd.DataFrame(grade_table)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(g["grade"], g["observed_rate"] * 100, color=[GRADES[x] for x in g["grade"]], width=0.6)
    for b, r, n in zip(bars, g["observed_rate"], g["companies"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{r * 100:.2f}%\n(n={n:,})", ha="center", va="bottom", color=INK2, fontsize=8.5)
    ax.set(ylabel="Observed 12-month insolvency rate (%)", xlabel="Risk grade")
    ax.set_ylim(0, g["observed_rate"].max() * 100 * 1.3)
    ax.grid(axis="x", visible=False)
    ax.set_title("Insolvency rate by grade (test month)")
    save(fig, "grade_rates.png")


def calibration(decile_table: list[dict]):
    d = pd.DataFrame(decile_table)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    top = max(d["predicted_rate"].max(), d["observed_rate"].max()) * 100 * 1.1
    ax.plot([0, top], [0, top], color=MUTED, lw=1, ls=":", label="Perfect calibration")
    ax.plot(d["predicted_rate"] * 100, d["observed_rate"] * 100, color=SERIES[0], lw=2, marker="o", ms=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="Risk deciles")
    ax.set(xlabel="Predicted PD (%)", ylabel="Observed rate (%)", xlim=(0, top), ylim=(0, top))
    ax.set_title("Calibration on the test month")
    ax.legend(loc="upper left")
    save(fig, "calibration.png")


def importance(fi: list[dict], k: int = 12):
    f = pd.DataFrame(fi).head(k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(f["feature"], f["mean_abs_shap"], color=SERIES[0], height=0.6)
    ax.set(xlabel="Mean |SHAP| (log-odds)")
    ax.grid(axis="y", visible=False)
    ax.set_title("What drives the insolvency score")
    save(fig, "shap_importance.png")


def sectors():
    p = PUB / "mart_register_health_by_sector.parquet"
    if not p.exists():
        return
    s = pd.read_parquet(p)
    s = s[~s["sic_section"].isin(["?", "Z", "T", "U"])].sort_values("insolvency_rate_pct")
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    import textwrap

    names = [textwrap.shorten(n, 42, placeholder="…") for n in s["sic_section_name"]]
    ax.barh(names, s["insolvency_rate_pct"], color=SERIES[0], height=0.6)
    ax.set(xlabel="% of live companies in insolvency proceedings")
    ax.grid(axis="y", visible=False)
    ax.set_title("Share of live companies in insolvency, by sector")
    save(fig, "sector_insolvency.png")


def main() -> None:
    metrics = json.loads((REPORTS / "metrics.json").read_text())
    ins = metrics["models"]["insolvency"]
    preds = pd.read_parquet(REPORTS / "test_predictions_insolvency.parquet")
    capture_curve(preds, "label_insolvency")
    grade_rates(ins["grade_table"])
    calibration(ins["decile_table"])
    importance(ins["feature_importance"])
    sectors()
    print("figures written to", FIG, file=sys.stderr)


if __name__ == "__main__":
    main()
