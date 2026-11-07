# Industrial Predictive Maintenance | Leakage-Safe ML Research Benchmark

An applied manufacturing ML pipeline for forecasting risk of machine failure from simulated process measurements. This portfolio implementation emphasizes **valid evaluation**, **risk-sensitive thresholds**, and **transparent operational limitations**, rather than inflated accuracy claims.

## What it implements
- Schema and class distribution validation for machine telemetry.
- Drops five post-outcome fault-mode labels, Product ID and UID to avoid leakage.
- Stratified **60/20/20 train-validation-test**, training-only preprocessing.
- Class-weighted logistic regression vs histogram gradient boosting.
- Validation-only threshold optimization using illustrative asymmetric false-negative/false-positive costs.
- Unsupervised Isolation Forest anomaly ranking using healthy training observations.
- Held-out precision, recall, F2, PR-AUC, ROC-AUC and confusion matrix.
- Machine-readable model card, results summary and automated tests.

## Data provenance
Download the publicly available **AI4I 2020 Predictive Maintenance Dataset**, from UCI: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Author: S. Matzka, 2020. DOI: https://doi.org/10.24432/C5HS5C. License: CC BY 4.0. **This dataset is intentionally simulated to resemble industrial signals; it is NOT recorded factory sensor telemetry.** Data are not bundled in the repository.

## Run
```bash
pip install -r requirements.txt
python maintenance.py --csv path/to/ai4i2020.csv --output reports
pytest -q
```

## Technical caveats
This dataset has no machine event timestamps, so random stratification is *not* proof of generalization across time or to unseen industrial assets. Fault-mode columns identify the outcome and must not be used as predictors. False-alarm and missed-failure costs (1 and 8) are **illustrative**, not factory-validated economics. Do not use this benchmark as a live maintenance decision tool.

## Proposed follow-on engineering
Gather permissioned asset-level sensor logs and annotated failure times; create lag/rolling-window features and leakage-controlled asset/time holdouts; evaluate detection lead-time, calibration, false alarms per asset-week, severity-based priorities and process drift; build an operator-facing monitoring dashboard. Those extensions are **not** claimed as completed here.

This is original educational code, not an industrial deployment.
