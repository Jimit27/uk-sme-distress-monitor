"""UK SME Distress Monitor - Streamlit dashboard.

Reads only the small files in data/published/, which CI refreshes. Run with:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PUB = Path(__file__).resolve().parents[1] / "data" / "published"

# Palette (validated reference instance): chrome, categorical slots, ordinal blue ramp.
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
GRADE_COLORS = {"A": "#86b6ef", "B": "#5598e7", "C": "#2a78d6", "D": "#1c5cab", "E": "#104281"}
GRADE_WORDS = {"A": "Low", "B": "Below average", "C": "Elevated", "D": "High", "E": "Very high"}
MODEL_NAMES = {
    "rule_negative_equity": "Rule: negative equity",
    "logistic_regression": "Logistic regression",
    "lightgbm_calibrated": "LightGBM (monotone, calibrated)",
}

st.set_page_config(page_title="UK SME Distress Monitor", page_icon="📉", layout="wide")


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=13),
        title_font=dict(color=INK, size=15),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color=INK2)),
        hoverlabel=dict(bgcolor="white", font_color=INK),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED), zeroline=False)
    return fig


@st.cache_data
def load_json(name: str) -> dict:
    p = PUB / name
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data
def load_parquet(name: str) -> pd.DataFrame:
    p = PUB / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def pct(x: float, dp: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{dp}f}%"


def money(v) -> str:
    if v is None or pd.isna(v):
        return "n/a"
    s = "-" if v < 0 else ""
    v = abs(v)
    return f"{s}£{v / 1e6:.2f}m" if v >= 1e6 else (f"{s}£{v / 1e3:.0f}k" if v >= 1e3 else f"{s}£{v:.0f}")


metrics = load_json("metrics.json")
register = load_json("register_summary.json")
live = load_parquet("live_scores.parquet")
ins = metrics.get("models", {}).get("insolvency", {})
best = ins.get("test", {}).get("lightgbm_calibrated", {})

st.title("UK SME Distress Monitor")
st.caption(
    "Early-warning scores for UK companies, built from the accounts they file at Companies House. "
    "The model predicts which companies enter formal insolvency within 12 months of filing, "
    "trained and tested on real filings and real outcomes."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies on the register", f"{register.get('companies', 0):,}")
c2.metric("In insolvency proceedings", f"{register.get('in_insolvency', 0):,}")
c3.metric("Out-of-time test AUC", f"{best.get('roc_auc', float('nan')):.3f}" if best else "n/a")
c4.metric("Insolvencies caught in riskiest 10%", pct(best.get("capture_top_10pct")) if best else "n/a")

tab_watch, tab_model, tab_register, tab_method = st.tabs(["Watchlist", "Model performance", "Register health", "Method"])

# ---------------------------------------------------------------- watchlist
with tab_watch:
    if live.empty:
        st.info("No live scores published yet. The daily workflow scores the newest filings each weekday.")
    else:
        dates = live["filing_batch"].astype(str)
        st.markdown(
            f"**{len(live):,} trading companies** that filed accounts between **{dates.min()}** and **{dates.max()}**, "
            "ranked by estimated 12-month insolvency probability."
        )
        f1, f2, f3 = st.columns([1, 2, 2])
        grades = f1.multiselect("Grade", list("ABCDE"), default=["D", "E"])
        sectors = sorted(live["sic_section_name"].dropna().unique())
        sector = f2.multiselect("Sector", sectors)
        query = f3.text_input("Search name or company number")
        view = live[live["grade"].isin(grades)] if grades else live
        if sector:
            view = view[view["sic_section_name"].isin(sector)]
        if query:
            q = query.strip().upper()
            view = view[view["entity_name"].fillna("").str.upper().str.contains(q, regex=False) | (view["company_number"] == q)]

        table = view.assign(
            **{
                "PD (12m)": view["pd"],
                "Assets": view["total_assets"].map(money),
                "Net assets": view["equity"].map(money),
            }
        )[["company_number", "entity_name", "grade", "PD (12m)", "Assets", "Net assets", "sic_section_name", "post_town", "filing_batch"]]
        table.columns = ["Number", "Company", "Grade", "PD (12m)", "Assets", "Net assets", "Sector", "Town", "Filed"]
        st.dataframe(
            table.head(500),
            hide_index=True,
            use_container_width=True,
            column_config={"PD (12m)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0, max_value=float(max(0.01, live["pd"].max())))},
        )
        if len(view) > 500:
            st.caption(f"Showing the riskiest 500 of {len(view):,} matching companies.")

        st.subheader("Company view")
        options = view["company_number"].head(500).tolist()
        if options:
            labels = {r.company_number: f"{r.entity_name or r.company_number} ({r.company_number})" for r in view.head(500).itertuples()}
            pick = st.selectbox("Company", options, format_func=lambda n: labels.get(n, n))
            row = live.set_index("company_number").loc[pick]
            a, b = st.columns([1, 2])
            a.metric("Grade", f"{row['grade']} · {GRADE_WORDS[row['grade']]}")
            a.metric("12-month insolvency probability", pct(row["pd"]))
            a.metric("12-month failure/exit probability", pct(row["pd_failure"]))
            b.markdown(f"> {row['summary']}")
            figures = pd.DataFrame(
                {
                    "Total assets": [row.get("total_assets")],
                    "Net assets": [row.get("equity")],
                    "Net assets (prior year)": [row.get("equity_prior")],
                    "Cash": [row.get("cash")],
                    "Current assets": [row.get("current_assets")],
                    "Current liabilities": [row.get("current_liabilities")],
                }
            ).T.rename(columns={0: "Value"})
            figures["Value"] = figures["Value"].map(money)
            b.table(figures)
            b.caption(f"Companies House: https://find-and-update.company-information.service.gov.uk/company/{pick}")

# ------------------------------------------------------------ model performance
with tab_model:
    if not ins:
        st.info("Model metrics not published yet.")
    else:
        cohort = metrics["cohort"]
        st.markdown(
            f"Trained on accounts filed in **{', '.join(ins['train_batches'])}** and tested out-of-time on "
            f"**{', '.join(ins['test_batches'])}** filings ({best['n']:,} companies, {best['events']:,} insolvencies, "
            f"base rate {pct(best['base_rate'], 2)}). Outcomes read from the register about 12 months later."
        )
        rows = []
        for key, name in MODEL_NAMES.items():
            m = ins["test"][key]
            rows.append(
                {
                    "Model": name,
                    "ROC AUC": m["roc_auc"],
                    "Gini": m["gini"],
                    "PR AUC": m["pr_auc"],
                    "KS": m["ks"],
                    "Capture top 5%": m["capture_top_5pct"],
                    "Capture top 10%": m["capture_top_10pct"],
                    "Lift top 1%": m["lift_top_1pct"],
                }
            )
        st.dataframe(
            pd.DataFrame(rows).style.format({c: "{:.3f}" for c in ["ROC AUC", "Gini", "PR AUC", "KS"]} | {"Capture top 5%": "{:.1%}", "Capture top 10%": "{:.1%}", "Lift top 1%": "{:.1f}x"}),
            hide_index=True,
            use_container_width=True,
        )

        preds = load_parquet("test_predictions_insolvency.parquet")
        left, right = st.columns(2)
        if not preds.empty:
            y = preds["label_insolvency"].to_numpy()
            fig = go.Figure()
            order = np.argsort(-preds["pd"].to_numpy(), kind="stable")
            share = np.arange(1, len(y) + 1) / len(y)
            fig.add_trace(go.Scatter(x=share * 100, y=np.cumsum(y[order]) / y.sum() * 100, name="LightGBM", line=dict(color=SERIES[0], width=2),
                                     hovertemplate="Riskiest %{x:.1f}% of companies<br>%{y:.1f}% of insolvencies<extra></extra>"))
            order_l = np.argsort(-preds["pd_logistic"].to_numpy(), kind="stable")
            fig.add_trace(go.Scatter(x=share * 100, y=np.cumsum(y[order_l]) / y.sum() * 100, name="Logistic regression", line=dict(color=SERIES[1], width=2),
                                     hovertemplate="Riskiest %{x:.1f}% of companies<br>%{y:.1f}% of insolvencies<extra></extra>"))
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Random", line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip"))
            fig.update_layout(title="Share of insolvencies caught vs share of companies reviewed")
            fig.update_xaxes(title="Companies reviewed, riskiest first (%)", range=[0, 100])
            fig.update_yaxes(title="Insolvencies caught (%)", range=[0, 100])
            left.plotly_chart(style(fig), use_container_width=True)

        grade_tbl = pd.DataFrame(ins["grade_table"])
        fig = go.Figure(
            go.Bar(
                x=grade_tbl["grade"],
                y=grade_tbl["observed_rate"] * 100,
                marker=dict(color=[GRADE_COLORS[g] for g in grade_tbl["grade"]], cornerradius=4),
                customdata=np.stack([grade_tbl["companies"], grade_tbl["events"], grade_tbl["mean_pd"] * 100], axis=1),
                hovertemplate="Grade %{x}<br>Observed insolvency rate %{y:.2f}%<br>Predicted %{customdata[2]:.2f}%<br>%{customdata[1]} of %{customdata[0]:,} companies<extra></extra>",
                text=[f"{v:.1f}%" for v in grade_tbl["observed_rate"] * 100],
                textposition="outside",
                textfont=dict(color=INK2),
            )
        )
        fig.update_layout(title="Observed 12-month insolvency rate by grade (test set)", bargap=0.35, showlegend=False)
        fig.update_yaxes(title="Insolvency rate (%)")
        right.plotly_chart(style(fig), use_container_width=True)

        dec = pd.DataFrame(ins["decile_table"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dec["predicted_rate"] * 100, y=dec["observed_rate"] * 100, mode="markers+lines", name="Deciles",
                                 marker=dict(size=9, color=SERIES[0], line=dict(color=SURFACE, width=2)), line=dict(color=SERIES[0], width=2),
                                 hovertemplate="Decile %{text}<br>Predicted %{x:.2f}%<br>Observed %{y:.2f}%<extra></extra>", text=dec["decile"]))
        top = float(max(dec["predicted_rate"].max(), dec["observed_rate"].max()) * 100 * 1.1)
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], name="Perfect calibration", line=dict(color=MUTED, width=1, dash="dot"), hoverinfo="skip"))
        fig.update_layout(title="Calibration: predicted vs observed by risk decile")
        fig.update_xaxes(title="Predicted PD (%)")
        fig.update_yaxes(title="Observed rate (%)")
        left2, right2 = st.columns(2)
        left2.plotly_chart(style(fig), use_container_width=True)

        imp = pd.DataFrame(ins["feature_importance"]).head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=imp["mean_abs_shap"], y=imp["feature"], orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                               hovertemplate="%{y}<br>mean |SHAP| %{x:.3f}<extra></extra>"))
        fig.update_layout(title="What drives the score (mean |SHAP|, test set)", showlegend=False, bargap=0.3)
        right2.plotly_chart(style(fig, 400), use_container_width=True)

# ------------------------------------------------------------ register health
with tab_register:
    sector = load_parquet("mart_register_health_by_sector.parquet")
    area = load_parquet("mart_register_health_by_area.parquet")
    vintage = load_parquet("mart_incorporations_by_year.parquet")
    if sector.empty:
        st.info("Register marts not published yet.")
    else:
        s = sector[~sector["sic_section"].isin(["?", "Z", "T", "U"])].sort_values("insolvency_rate_pct")
        fig = go.Figure(go.Bar(x=s["insolvency_rate_pct"], y=s["sic_section_name"], orientation="h", marker=dict(color=SERIES[0], cornerradius=4),
                               customdata=np.stack([s["companies"], s["in_insolvency"]], axis=1),
                               hovertemplate="%{y}<br>%{x:.2f}% in insolvency<br>%{customdata[1]:,} of %{customdata[0]:,} companies<extra></extra>"))
        fig.update_layout(title="Share of live companies in insolvency proceedings, by sector", showlegend=False, bargap=0.3)
        fig.update_xaxes(title="% of companies on the register")
        st.plotly_chart(style(fig, 560), use_container_width=True)

        a, b = st.columns(2)
        if not area.empty:
            top = area.sort_values("insolvency_rate_pct", ascending=False).head(15)
            a.markdown("**Postcode areas with the highest insolvency share** (areas with 1,000+ companies)")
            a.dataframe(
                top[["postcode_area", "main_town", "companies", "insolvency_rate_pct", "overdue_rate_pct"]].rename(
                    columns={"postcode_area": "Area", "main_town": "Main town", "companies": "Companies", "insolvency_rate_pct": "Insolvency %", "overdue_rate_pct": "Accounts overdue %"}
                ),
                hide_index=True,
                use_container_width=True,
            )
        if not vintage.empty:
            v = vintage[vintage["incorporation_year"] <= vintage["incorporation_year"].max() - 1]
            fig = go.Figure(go.Scatter(x=v["incorporation_year"], y=v["insolvency_rate_pct"], mode="lines", line=dict(color=SERIES[0], width=2),
                                       hovertemplate="Incorporated %{x}<br>%{y:.2f}% in insolvency<extra></extra>"))
            fig.update_layout(title="Insolvency share by year of incorporation", showlegend=False)
            fig.update_yaxes(title="% in insolvency")
            b.plotly_chart(style(fig), use_container_width=True)

# ------------------------------------------------------------ method
with tab_method:
    st.markdown(
        """
**Data.** Companies House publishes every electronically filed set of accounts as inline XBRL, plus a monthly
snapshot of every live company on the register. Both are free bulk downloads.

**Point-in-time design.** Features come only from accounts filed in the training and test months. Each
company's outcome is read from the register snapshot about 12 months later:

* *insolvency* - the company is in liquidation, administration, receivership or a CVA;
* *failure* - insolvency, a pending proposal to strike off, or the company has already left the register.

Nothing the model sees is dated after the filing. The company's sector and location come from the *current*
register, so they are shown in the dashboard but never used by the model, because dissolved companies would
be missing them and that gap would leak the outcome. Company age is estimated from the company number (numbers
are issued in sequence) using a neighbouring company, for the same reason.

**Models.** A logistic regression benchmark and a LightGBM model with monotone constraints on the textbook
credit relationships (more equity means less risk, late filing means more risk), calibrated with isotonic
regression. Tested out of time on the following month's filers. Every score comes with its top three SHAP drivers.

**Limits.** Only about 75% of accounts are filed electronically. Micro-entity accounts carry few figures, and
there is no profit and loss account for most small companies. Companies that went through a fast liquidation
*and* dissolved within the year appear as "removed", so they count towards failure but not insolvency, which
slightly understates the insolvency label. A score is a statistical signal, not a credit decision.
"""
    )
