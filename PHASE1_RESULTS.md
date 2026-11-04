# ForeSightAI — Phase 1 recorded results

Human-readable summary of the modeling evaluation. Every number below comes
from an existing modeling report; nothing is estimated. Both reports were
produced by the leakage-safe protocol: preprocessing statistics are fit on
TRAIN only, hyperparameters are chosen by VALIDATION MAE only, and TEST is
scored once as a final holdout after selection is frozen.

- **V2 figures** — `data/processed/modeling_report_v2.json`,
  recorded run `2026-09-28T04:16:53Z`. This report was independently verified
  to be byte-equivalent to a fresh `run_demo.py` execution, so it is current.
- **V1 figures** — a fresh `python run_demo.py` execution
  (`2026-10-10T20:58:10Z`), because the canonical V1 report
  `data/processed/modeling_report.json` is **stale**: it predates the Elastic
   Net model and does not contain an `elastic_net` entry. Do not cite the
   canonical V1 report for Elastic Net. The demo runner regenerates a
   consistent V1 report on demand.

> **Report/source metadata version mismatch (see `README.md`):** the committed
> reports in `data/processed/` were generated from earlier `run_modeling.py`
> revisions and were **intentionally preserved unchanged** (not regenerated)
> during this documentation correction. `modeling_report.json` is an older V1
> report (recorded run `2026-08-27T06:05:24Z`) that predates the current Elastic
> Net implementation and reporting behavior; `modeling_report_v2.json` records
> results generated at `2026-09-28T04:16:53Z`. The current `run_modeling.py`
> emits a `feature_set_comparison` field, whereas both committed reports contain
> the older `feature_set_winner_rule` field. This is a metadata version mismatch
> only: the recorded numerical metrics have not been independently re-run
> against the current source and are neither confirmed nor changed by it.

Splits are identical for both feature sets (chronological):
train 4580 / validation 1008 / test 910 rows, `target_return_1d`.

## V2 feature set (16 stationary features)

| model | validation MAE | test MAE | test R² |
|---|---|---|---|
| zero_return | 0.009367 | 0.006582 | −0.007422 |
| train_mean | 0.009353 | 0.006562 | −0.004144 |
| ridge (α=0.01, validation-selected) | 0.009377 | 0.006642 | −0.009550 |
| elastic_net (α=0.01, l1_ratio=0.1, validation-selected) | 0.009353 | 0.006562 | −0.004144 |

## V1 feature set (10 features)

| model | validation MAE | test MAE | test R² |
|---|---|---|---|
| zero_return | 0.009367 | 0.006582 | −0.007422 |
| train_mean | 0.009353 | 0.006562 | −0.004144 |
| persistence | 0.013913 | 0.009602 | −1.079653 |
| ridge (α=0.01, validation-selected) | 0.009348 | 0.006768 | −0.062570 |
| elastic_net (α=0.01, l1_ratio=0.1, validation-selected) | 0.009353 | 0.006562 | −0.004144 |

## What these numbers show

- **No model demonstrates an out-of-sample improvement over the train-mean
  baseline.** On TEST, `train_mean` has the lowest MAE (0.006562) and the
  least-negative R² (−0.004144) in both feature sets. The tuned regression
  models do not beat it.
- **Elastic Net is degenerate under the canonical grid.** It selects
  `α=0.01, l1_ratio=0.1` and shrinks every coefficient to exactly zero, so it
  is numerically identical to the `train_mean` baseline. This is documented in
  `context.txt` (§22) as a known open issue.
- **Ridge is at or below the baseline.** V1 ridge has a marginally lower
  validation MAE (0.009348 vs 0.009353) but a materially worse TEST MAE
  (0.006768) and R² (−0.0626). V2 ridge is worse than the baseline on both
  validation and test.
- **`persistence` is clearly worse** (TEST R² −1.08), confirming the
  evaluation discriminates between models.
- Test R² is slightly negative for all models because the TEST period has a
  smaller target variance than TRAIN; predicting the train mean therefore
  loses to the test-mean oracle, which is expected for a near-unpredictable
  daily return.

**Conclusion:** the pipeline is leakage-safe and reproducible, but the
current models do not demonstrate predictive skill. No claim of reliable
stock-market prediction is supported by these measurements.

## Scope not covered by these results

The external-data integration scaffolding (`backend/pipeline/external/`), the
legacy RAG chatbot prototype (`backend/main.py`), and the frontend
(`frontend/src/`, source only, no build) are **not** part of the runnable
Phase 1 forecasting demo and are not evaluated by the numbers above. They are
isolated, prototype, or future work; no external-data or chatbot result is
claimed here.
