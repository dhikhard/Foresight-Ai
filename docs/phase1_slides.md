# Slide 1 — Title

**ForeSightAI: AI-Driven Probabilistic Forecasting Engine**

Phase 1 Project Presentation

- Project team: Chahel A (USN: NNM23CC012); Hardhik Taranath Shetty (USN: NNM23CC017); Poorvika (USN: NNM23CC038)
- Department: Computer and Communication Engineering
- Institution: NMAM Institute of Technology
- Guide: Prof. Chidananda T

# Slide 2 — Problem Statement

Financial market returns are difficult to forecast because they are noisy and influenced by many changing factors.

The objective of ForeSightAI is to develop a systematic forecasting pipeline that processes historical market data, engineers relevant features, trains regression models, and evaluates their performance against simple baselines.

**Problem:** Can machine learning models produce useful next-day S&P 500 return forecasts beyond simple baseline predictions?

# Slide 3 — Objectives and Scope

**Objectives**
- Process historical S&P 500 market data.
- Generate features from historical price information.
- Train and evaluate regression models.
- Prevent data leakage during preprocessing and model evaluation.
- Compare learned models with simple baseline predictions.
- Produce reproducible reports and test results.

**Phase 1 scope:** Data processing, feature engineering, baseline modeling, Ridge Regression, Elastic Net, evaluation, and automated testing.

**Not in Phase 1:** external-data integration into the forecasting dataset, the dashboard, and the chatbot.

# Slide 4 — System Architecture

Raw financial market data
↓
Data cleaning and preprocessing
↓
Feature engineering — V1 and V2
↓
Chronological train / validation / test split
↓
Training-only preprocessing
↓
Baseline models, Ridge Regression, and Elastic Net
↓
Validation-based model selection
↓
Final evaluation and report generation

The current Phase 1 deliverable is the data and modeling pipeline. External-data integration (isolated scaffolding, not integrated into the dataset), a production dashboard, and the chatbot (legacy prototype, requires an API key) are not part of the runnable Phase 1 verification.

# Slide 5 — Methodology and Leakage Prevention

- Historical S&P 500 data is processed chronologically.
- V1 and V2 provide separate feature pipelines.
- The V1 feature set contains 10 features.
- The V2 feature set contains 16 features.
- Training, validation, and test data are separated by time.
- Preprocessing parameters are learned from training data only.
- Validation data is used for model selection.
- The held-out test set is reserved for final evaluation.

**Dataset split:** 4,580 training rows, 1,008 validation rows, and 910 test rows for the V2 evaluation.

# Slide 6 — Models and Evaluation

**Models implemented**
- Zero-return baseline
- Train-mean baseline
- Persistence baseline (V1 only — the V2 feature set provides no same-day return column)
- Ridge Regression
- Elastic Net Regression

**Evaluation metrics**
- Mean Absolute Error (MAE)
- R² score
- Directional accuracy

Directional accuracy on the V2 test split is reported as a measured value: approximately 0.56 for the constant baselines and 0.51 for Ridge Regression. These figures are not compared against a directional baseline or against the test split's own direction distribution, so no directional skill is claimed.

The baselines are essential because a complex model is useful only if it provides an improvement over a simple alternative.

# Slide 7 — Results and Limitations

The recorded V2 test results (canonical run, `generated_at_utc` 2026-09-28T04:16:53Z) include:

| Model | Test MAE | Test R² |
|---|---|---|
| Train-mean baseline | 0.006562 | −0.0041 |
| Ridge Regression | 0.006642 | −0.0096 |
| Elastic Net | 0.006562 | −0.0041 |

Negative R² indicates that these predictions have higher squared error than a predictor using the test-set mean.

The canonical Elastic Net configuration currently produces zero-valued coefficients, making its predictions equivalent to the train-mean baseline.

**Finding:** The current experiments do not demonstrate improved out-of-sample predictive performance over the train-mean baseline.

**Report metadata note:** the committed JSON reports predate the current `run_modeling.py` reporting and contain the older `feature_set_winner_rule` field, whereas the current code emits `feature_set_comparison`. This is a metadata mismatch only and does not change the numbers above (see preparation notes).

# Slide 8 — Conclusion and Future Work

**Completed**
- Historical data-processing and feature-engineering pipelines.
- V1 and V2 modeling workflows.
- Baseline, Ridge, and Elastic Net implementations.
- Leakage-aware evaluation.
- Automated tests (246 passing) and a reproducible demonstration runner.
- Documentation and recorded experimental results.

**Future work**
- Introduce an Elastic-Net-specific alpha grid and re-evaluate using validation-only selection.
- Explore walk-forward evaluation instead of a single split.
- Investigate additional features and probabilistic calibration.
- Integrate external data sources into the dataset (currently isolated scaffolding, not evaluated), and develop the dashboard/chatbot only after validating the forecasting methodology.

**Conclusion:** ForeSightAI establishes a reproducible forecasting and evaluation pipeline. Further work is required to establish genuine predictive skill.

---

## Preparation notes (not part of the 8 slides)

- **Metric source:** `data/processed/modeling_report_v2.json`, `feature_set=v2`,
  `target=target_return_1d`, `generated_at_utc=2026-09-28T04:16:53.913482+00:00`.
  Verified values (full precision): train-mean test MAE `0.00656153`, R²
  `−0.00414437`; Ridge test MAE `0.00664184`, R² `−0.00955029`; Elastic Net test
  MAE `0.00656153`, R² `−0.00414437` (equals train-mean). Test directional
  accuracy: constant baselines `0.5604`, Ridge `0.5066`. Rows: train 4580 /
  validation 1008 / test 910.
- **Report/source metadata mismatch:** the committed reports
  `data/processed/modeling_report.json` (older V1 report, run
  `2026-08-27T06:05:24Z`, no `elastic_net` entry) and
  `data/processed/modeling_report_v2.json` (run `2026-09-28T04:16:53Z`) were
  generated from earlier `run_modeling.py` revisions and were **intentionally
  preserved unchanged** (not regenerated) during this documentation correction.
  They contain the older `feature_set_winner_rule` field, whereas the current
  `run_modeling.py` emits `feature_set_comparison`. This is a metadata-only
  mismatch: the recorded metrics were not independently re-run against the
  current source and are neither confirmed nor changed by it.
- **Architecture (Slide 4)** is a text flowchart; convert to a diagram object
  when producing the `.pptx`.
- **Team details (Slide 1)** are reproduced verbatim from the supplied draft;
  confirm spelling of names and the guide's name/title before submission.
