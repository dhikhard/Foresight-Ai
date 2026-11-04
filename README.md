# ForeSightAI

An auditable, leakage-safe S&P 500 market-data forecasting pipeline (Phase 1),
with self-contained NumPy models and rigorous temporal-safety guarantees.

> **Phase 1 demonstration = the market-data forecasting pipeline** described
> below: deterministic preprocessing, V1/V2 feature sets, chronological
> train/validation/test splits, train-only preprocessing, validation-only
> model selection, and baseline + linear models scored once on a held-out
> test period.
>
> **Headline finding:** the current models do **not** demonstrate an
> out-of-sample improvement over the train-mean baseline. See
> [Current results and limitations](#current-results-and-limitations) and
> [`PHASE1_RESULTS.md`](PHASE1_RESULTS.md). No claim of reliable
> stock-market prediction is supported.

A separate, earlier **RAG prototype** (`backend/main.py`) also exists; it is
not part of the Phase 1 demo and is documented at the end.

---

## Objective and scope

**Objective:** rebuild ForeSightAI as an auditable, leakage-safe market
forecasting system. Phase 1 builds the deterministic data pipeline and an
honest baseline/linear modeling layer *before* adding richer signals.

**In scope (implemented):**
- Deterministic raw→processed market-data pipeline (ingest, clean, V1 and V2
  features, forward targets, chronological splits) with byte-reproducible
  outputs.
- Leakage-safe modeling: train-only standardization, rule baselines, and
  closed-form Ridge / coordinate-descent Elastic Net (NumPy only).
- A full test suite (unit + integration) and a one-command demo runner.

**Out of scope (not implemented):** walk-forward validation, probability
calibration, neural/deep models, the dashboard/chat UI, and external data
integration into the forecasting dataset.

## Prerequisites and dataset

- **Python 3.10+** (developed and tested on **Python 3.14**).
- **Raw dataset (required):** the Kaggle *Global Financial Markets* CSV/zip.
  Place it in `data/raw/`. The current source file is
  `data/raw/global_financial_markets_2000_Now.csv`
  (≈189,787 rows, 34 symbols, 2000-01-03 → 2026-08-20). The raw file is only
  ever read; it is never modified.
- `data/raw/` and `data/processed/*.csv` are **gitignored**, so a fresh clone
  must supply the raw dataset and regenerate the processed CSVs. The committed
  `data/processed/*.json` reports are small provenance artifacts.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt   # pytest, for the tests
```

The market pipeline and modeling layer need only **numpy** (and the standard
library). The other entries in `backend/requirements.txt` support the optional
RAG prototype.

## Interactive Quantitative Dashboard (Live UI Prototype)

Launch the complete quantitative-finance dashboard with a single command:

```powershell
.\.venv\Scripts\python.exe run_ui.py
# or
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Once running, navigate to **http://localhost:8000** in your web browser.

The dashboard integrates directly with the verified market dataset (`sp500_clean.csv`, `sp500_features_v2.csv`), models (`RidgeRegressor`, `ElasticNetRegressor`, `TrainMeanBaseline`, `ZeroBaseline`), and reports (`modeling_report_v2.json`):

1. **Market Overview** (`/?tab=market-overview`):
   - S&P 500 historical price chart with SMA 20, 50, and 200 overlays and 1Y/3Y/5Y/All time filters.
   - Latest available observation metrics: close ($7,641.16 on 2026-08-20), daily return (-0.87%), volume, and day range.
   - Volatility indicators: 20-day annualized volatility (13.1%), 60-day annualized volatility (13.7%), and regime ratio.
   - Drawdown indicators: running all-time peak close ($7,806.60), current drawdown (-2.02%), and 52-week high/low range.
   - V2 stationary feature table with values, categories, mathematical descriptions, and status badges.
   - Data freshness provenance: verified audit against Kaggle raw source through August 20, 2026.

2. **Forecasting Engine** (`/?tab=forecasting-engine`):
   - Computes honest next-day return predictions using the trained models on the latest observation.
   - Displays predicted percentage return and corresponding implied next-day S&P 500 index close ($7,650.58 for Ridge vs. $7,642.71 for Train-Mean).
   - Side-by-side comparison with Train-Mean baseline (+0.020%), Elastic Net (+0.020%), and Zero-Return null baseline (0.000%).
   - Explainability: feature attribution breakdown chart ($w_j \cdot \tilde{x}_j$) showing positive and negative drivers.
   - Explicit scientific disclosure: clearly explains that daily stock returns exhibit near-zero autocorrelation and linear models do not outperform the train-mean baseline on unseen future data.

3. **Model Performance** (`/?tab=model-performance`):
   - Direct comparison matrix across Train (< 2019), Validation (2019–2022), and Test (2023–2026) splits.
   - Exact metrics: MAE, RMSE, R², and Directional Accuracy.
   - Actual vs. predicted return chart covering all 910 held-out test days (Jan 2023 – Aug 2026).
   - Residual series plot and residual error distribution histogram.
   - Mathematical notes on why test R² is slightly negative.

4. **Risk Analysis & Stress Testing** (`/?tab=risk-analysis`):
   - User-entered hypothetical portfolio capital (e.g. $100,000, with quick chips: $10k, $50k, $100k, $1M).
   - Interactive market shocks: -10%, -5%, -3%, +3%, +5%, plus continuous custom shock slider (-20% to +20%).
   - Instant calculation of hypothetical dollar P&L and resulting portfolio equity.
   - Empirical non-parametric Value-at-Risk (95% VaR: -1.84%, 99% VaR: -3.41%) and Expected Shortfall (CVaR).
   - Historical drawdown underwater chart (2000–2026) and crisis benchmark impact table (Dot-Com -49.1%, 2008 GFC -56.8%, 2020 COVID -33.9%, 2022 Inflation -25.4%).
   - Prominent notice clearly identifying scenario simulations as hypothetical stress tests, not model predictions.

---

## Quick start — one-command demo runner (CLI)

Run the supported workflow (preprocess → V1 model → V2 features → V2 model)
into a single output directory. Everything is written **only** to that
directory; the canonical `data/processed` artifacts are never touched.

```powershell
# default: writes into a fresh temporary directory
.\.venv\Scripts\python.exe run_demo.py

# explicit output directory
.\.venv\Scripts\python.exe run_demo.py --out-dir demo_out

# use a specific raw dataset file
.\.venv\Scripts\python.exe run_demo.py --raw-path path\to\raw.csv --out-dir demo_out
```

- The runner refuses a **non-empty** `--out-dir` unless `--force` is given, so
  it cannot silently overwrite user files.
- It exits non-zero if any step fails.
- It writes a human-readable `demo_results.md` alongside the reports.
- No network calls and no API keys are used.

## Manual end-to-end reproduction (repository root)

All commands below were verified from the repository root.

```powershell
# 1) raw -> clean, V1 features, targets, and chronological split CSVs
.\.venv\Scripts\python.exe -m backend.pipeline.run_preprocess --with-targets

# 2) V2 features from the canonical cleaned series
.\.venv\Scripts\python.exe -m backend.pipeline.run_features_v2

# 3) V1 model + report (writes data/processed/modeling_report.json)
.\.venv\Scripts\python.exe -m backend.modeling.run_modeling

# 4) V2 model + report (writes data/processed/modeling_report_v2.json)
.\.venv\Scripts\python.exe -m backend.modeling.run_modeling --feature-set v2
```

`run_preprocess` options: `--symbol ^IXIC`, `--validation-start 2019-01-01`,
`--test-start 2023-01-01`, `--raw-path <file>`, `--processed-dir <dir>`.
`run_features_v2` options: `--clean-path <file>`, `--processed-dir <dir>`,
`--symbol`. `run_modeling` options: `--feature-set {v1,v2}`, `--features`,
`--targets`, `--out`.

> **Note on split boundaries:** `run_modeling` uses the default boundaries
> (`< 2019-01-01` / `2019-01-01–2022-12-31` / `≥ 2023-01-01`). If you override
> `--validation-start`/`--test-start` in `run_preprocess`, pass the same
> boundaries through the modeling configuration; the CLI does not forward
> them, so keep the defaults for the demo.

## Pipeline stages

### 1. Preprocessing (`backend/pipeline/run_preprocess.py`)
- Reads the raw CSV/zip, filters to `^GSPC` (S&P 500), sorts chronologically,
  deduplicates, and drops only rows with malformed dates or missing/non-numeric
  OHLC prices (each removal is logged). OHLC violations and negative prices are
  logged and **kept** (real history). Nothing is fabricated or forward-filled.
- Writes `sp500_clean.csv`, `sp500_features.csv`, `sp500_targets.csv` (with
  `--with-targets`), the three split CSVs, and `preprocessing_report.json`
  (including the raw-file SHA-256).

### 2. Feature Set V1 (10 features)
`return`, `log_return`, `ma_20`, `ma_50`, `ma_200`, `vol_20_annualized`,
`rolling_peak`, `drawdown`, `daily_range`, `open_close_return`. All are
backward-looking; warm-up rows are empty until enough history exists.

### 3. Feature Set V2 (16 stationary features)
`ret_lag_1/2/5/10/20`, `mom_5/10/20`, `ma_dist_20/50/200`,
`vol_60_annualized`, `vol_ratio_20_60`, `vol_change_5d`, `rsi_14`,
`bollinger_pctb_20`. V2 replaces price-level features with stationary ones.
Generated by `backend.pipeline.run_features_v2` from `sp500_clean.csv`
(byte-reproducible).

### 4. Targets
Forward-looking only (never model inputs): `target_return_1d/5d/10d` and
`target_future_vol_20`. Modeling uses `target_return_1d`. Tail rows have empty
targets and are dropped.

### 5. Chronological splits
Reused from the pipeline config:
- **train** `< 2019-01-01`
- **validation** `2019-01-01 ≤ d < 2023-01-01`
- **test** `≥ 2023-01-01`

After dropping warm-up (`ma_200` unavailable) and the final no-target row, the
modeling splits are train 4580 / validation 1008 / test 910 rows.

### 6. Leakage controls
- **Train-only preprocessing:** the standardizer is fit on TRAIN only; its
  frozen means/stds transform validation and test.
- **Validation-only selection:** Ridge `alpha` and the Elastic Net
  `(alpha, l1_ratio)` are chosen by VALIDATION MAE only.
- **Test scored once:** TEST is a final holdout, scored only after all
  selection decisions are frozen; it is never used for tuning.

## Models actually implemented

| Model | Rule | Inputs |
|---|---|---|
| `zero_return` | always predicts 0 | none |
| `train_mean` | predicts the constant TRAIN-mean target | none |
| `persistence` | predicts tomorrow = today's `return` (V1 only) | `return` column |
| `ridge` | closed-form ridge; `alpha` chosen by validation MAE | standardized features |
| `elastic_net` | NumPy coordinate descent; `(alpha, l1_ratio)` chosen by validation MAE | standardized features |

Canonical grids: `alpha ∈ {0.01, 0.1, 1, 10, 100}`,
`l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`. Elastic Net's objective is scaled by
`1/(2n)`, so its `alpha` grid is not on the same numerical scale as Ridge's
unnormalized grid. `persistence` is unavailable for V2 (no same-day `return`
column; `ret_lag_1` is yesterday's return and is not a persistence input).

## Outputs and reports

Written to `data/processed/` by the manual commands (or to `--out-dir` by
`run_demo.py`):

| File | Contents |
|---|---|
| `sp500_clean.csv` | cleaned, sorted OHLCV + metadata |
| `sp500_features.csv` | clean data + V1 features |
| `sp500_targets.csv` | forward-looking targets only (`--with-targets`) |
| `sp500_features_train/validation/test.csv` | chronological splits |
| `sp500_features_v2.csv` | date + 16 V2 features |
| `sp500_features_v2_report.json` | V2 provenance + warm-up audit |
| `preprocessing_report.json` | cleaning summary, gaps, config, raw SHA-256 |
| `modeling_report.json` | **V1** results (see report/source note below) |
| `modeling_report_v2.json` | **V2** results |
| `demo_results.md` | Markdown results table (demo runner only) |

**Report/source metadata version mismatch:** the committed reports in
`data/processed/` were generated from earlier revisions of `run_modeling.py`
and were **intentionally preserved unchanged** (not regenerated) during this
documentation correction:

- `modeling_report.json` is an older **V1** report (recorded run
  `2026-08-27T06:05:24Z`). It predates the current Elastic Net implementation
  and reporting behavior and does **not** contain an `elastic_net` entry.
- `modeling_report_v2.json` records results generated at
  `2026-09-28T04:16:53Z`.
- The current `backend/modeling/run_modeling.py` emits a
  `feature_set_comparison` field, whereas both committed reports contain the
  older `feature_set_winner_rule` field. (`run_modeling.py` no longer emits
  `feature_set_winner_rule`.)

This is a report/source **metadata** version mismatch and nothing more. The
recorded numerical metrics have not been independently re-run against the
current source, so this note neither confirms nor alters them. For a report
produced by the current code, run `run_demo.py` (which writes a fresh report to
its output directory) or re-run the manual V1/V2 modeling commands.

## Current results and limitations

Measured on the recorded runs (see [`PHASE1_RESULTS.md`](PHASE1_RESULTS.md)):

- **No model beats the train-mean baseline on the test period.** Test MAE:
  `train_mean` 0.006562; V1 `ridge` 0.006768; V2 `ridge` 0.006642;
  `elastic_net` 0.006562 (identical to `train_mean`). Test R² is slightly
  negative for every model.
- **Elastic Net is degenerate under the canonical grid:** it selects
  `alpha=0.01, l1_ratio=0.1` and shrinks all coefficients to zero, so it equals
  the constant train-mean predictor.
- Daily S&P 500 returns are close to unpredictable at this horizon, so the
  near-baseline results are the honest expected outcome at this stage.

**Therefore: do not claim reliable stock-market prediction.** The value of
Phase 1 is the reproducible, leakage-safe infrastructure, not predictive skill.

Next steps (not implemented): an Elastic-Net-specific alpha grid and/or a
walk-forward validation scheme, probability calibration (Brier score,
reliability curves), and richer signals.

## Testing

```powershell
# full suite (257 passing tests)
.\.venv\Scripts\python.exe -m pytest tests -q

# UI dashboard and quantitative engine tests
.\.venv\Scripts\python.exe -m pytest tests/test_ui_api.py -v

# focused areas
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py tests/test_features_v2.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_modeling.py tests/test_modeling_v2.py tests/test_elastic_net.py -q
.\.venv\Scripts\python.exe -m pytest tests/test_demo_runner.py -q
```

Covers ingestion, cleaning, exact feature math, warm-up handling,
look-ahead-leakage prevention, split boundaries, byte-for-byte reproducibility,
the feature/target contract, train-only preprocessing, validation-only
selection, Elastic Net convergence, dashboard API routes, risk stress testing, and the demo runner.

## Repository layout

```
backend/
  api/                 dashboard REST endpoints (market, forecast, performance, risk)
  static/              responsive quantitative dashboard UI (HTML5, Tailwind, Chart.js)
  services/
    quant_engine.py    analytical engine for metrics, inference, residuals, and VaR
    rag_engine.py      legacy RAG prototype (text embeddings & Groq LLM)
  pipeline/            market-data pipeline (ingest, clean, features V1/V2,
                       targets, splits, config, CLI runners)
  modeling/            dataset contract, train-only preprocessing, baselines,
                       ridge + elastic net, metrics, CLI runner
  pipeline/external/   experimental external-data scaffolding (NOT part of
                       the Phase 1 demo; not integrated into the dataset)
  main.py              FastAPI server mounting dashboard routes and static UI
run_ui.py              one-command browser launcher for the quantitative dashboard
run_demo.py            one-command Phase 1 CLI demonstration
docs/
  screenshots/         high-resolution screenshots of all 4 dashboard pages
  capture_screenshots.py  headless screenshot capture automation
tests/                 unit + integration test suite (257 passing tests)
context.txt            project continuity / state file
PHASE1_RESULTS.md      human-readable results table
```
backend/
  pipeline/            market-data pipeline (ingest, clean, features V1/V2,
                       targets, splits, config, CLI runners)
  modeling/            dataset contract, train-only preprocessing, baselines,
                       ridge + elastic net, metrics, CLI runner
  pipeline/external/   experimental external-data scaffolding (NOT part of
                       the Phase 1 demo; not integrated into the dataset)
  main.py, core/, services/, models/   legacy RAG prototype (see below)
frontend/src/          React source only (no build scaffolding yet)
tests/                 unit + integration tests
run_demo.py            one-command Phase 1 demonstration
context.txt            project continuity / state file
PHASE1_RESULTS.md      human-readable results table
```

## Legacy RAG prototype (not the Phase 1 demo)

`backend/main.py` is an earlier FastAPI prototype: ask a future-event
question, supply a crowd probability and reference articles, get an
`ai_probability`, reasoning, citations, and an `information_edge`. It embeds
text locally (SHA-256 feature hashing), retrieves the top chunks, and calls a
Groq LLM.

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Create `backend/.env` (gitignored) with `GROQ_API_KEY=...` (and optionally
`GROQ_MODEL`). Missing key → `/api/predict` returns HTTP 500 with instructions.
Interactive docs at http://localhost:8000/docs. This prototype requires an API
key and network access and is **separate** from the market-data Phase 1 demo.

## Roadmap & Status

1. ~~Frontend dashboard & quantitative UI~~ — **COMPLETED** (4 interactive pages: Market Overview, Forecasting Engine, Model Performance, Risk Analysis).
2. ~~S&P 500 market-data ingestion and preprocessing~~ — **COMPLETED** (deterministic, byte-reproducible, zero leakage).
3. ~~Inference API serving trained models~~ — **COMPLETED** (`/api/models/forecast`, `/api/market/overview`, `/api/risk/analysis`).
4. Walk-forward validation and an Elastic-Net-specific alpha grid (next phase).
5. Probability calibration and evaluation (Brier score, reliability curves) (next phase).
6. Non-linear models (LightGBM / CatBoost) and macroeconomic cross-asset signals (next phase).
7. Prediction-market comparison (e.g. Kalshi, Polymarket) — external layer integration (scaffolding in place).

## Security notes

- Never commit `.env` or API keys; `.gitignore` already excludes them.
- The RAG backend is unauthenticated by design for local development only —
  do not expose it publicly as-is.
