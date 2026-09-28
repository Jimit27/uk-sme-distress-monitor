# Model card: UK SME insolvency early-warning model

| | |
|---|---|
| **Task** | Rank UK trading companies by probability of entering formal insolvency (liquidation, administration, receivership, CVA) within ~12 months of filing accounts |
| **Model** | LightGBM (15 leaves, 586 trees, monotone constraints on 8 credit features) + isotonic calibration |
| **Benchmark** | L2 logistic regression on median-imputed, standardised features (+ missing indicators) |
| **Training data** | 208,196 non-dormant companies with positive total assets that filed accounts at Companies House in July 2025 (1,664 insolvencies) |
| **Test data** | 195,405 *different* companies that filed in August 2025 (1,496 insolvencies) - out-of-time |
| **Label source** | Companies House register snapshot, 1 September 2026 |
| **Features** | 29 point-in-time features from the filed accounts only: size, leverage, liquidity, year-on-year changes, filing lag, registration type, approximate age |

## Performance (test month)

| Metric | Value |
|---|---|
| ROC AUC | 0.812 (bootstrap 95% CI 0.802 to 0.822) |
| Gini | 0.625 |
| KS | 0.474 |
| PR AUC | 0.068 (base rate 0.0077) |
| Insolvencies in riskiest 1% / 5% / 10% | 13.9% / 35.4% / 47.9% |
| Brier score | 0.00737 |
| Mean predicted vs observed rate | 0.79% vs 0.77% |

In-sample AUC on the training fit is 0.915, against 0.812 out of time. Some of that gap is expected overfit.
The rest is the month-on-month shift that the out-of-time design exists to measure.

## Intended use

Portfolio monitoring and prioritisation, for example deciding which SME customers or suppliers a credit team
reviews first. The model is **not** for automated decisions about individual companies.

## Known limitations and risks

* **Coverage.** Only electronically filed accounts (~75%) are included. Micro-entity accounts are sparse,
  and there is almost no P&L data.
* **Label noise.** Companies that liquidated *and* dissolved inside the window are labelled "removed", not
  insolvent. A single outcome snapshot gives slightly different horizons for the two months.
* **Stability.** There is only one test month so far. Scores should be monitored with PSI on features and
  scores, and the model retrained as new monthly cohorts mature.
* **Fairness.** The model contains no personal data or protected characteristics. It scores companies, not
  people. Registration type (England & Wales, Scotland, Northern Ireland, LLP) is used as a feature. It has
  low importance, but it should be reviewed if the model is used for decisions.
