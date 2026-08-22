# ForesightAI

AI-driven probabilistic event forecasting engine — Phase 1 MVP.

Ask a future-event question, supply a crowd probability and reference articles, and get an
AI-estimated probability (`ai_probability`), reasoning, citations, and an `information_edge`
(the gap between the AI estimate and the crowd estimate).

## How It Works

```
Browser form (React)
   │  POST /api/predict  {event_question, manual_crowd_prob, context_articles[]}
   ▼
FastAPI backend
   ├─ chunk pasted articles (1200 chars / 200 overlap)
   ├─ embed with local SHA-256 feature hashing (384-dim, no external embedding API)
   ├─ retrieve top-5 relevant chunks (ephemeral in-memory ChromaDB per request)
   ▼
Groq LLM (llama-3.3-70b-versatile, temperature=0, JSON mode)
   ▼
{ai_probability, reasoning, citations} → information_edge = ai − crowd
```

## Project Structure

```
backend/
  main.py               FastAPI app: GET /health, POST /api/predict
  core/config.py        Settings from environment (.env supported)
  core/prompts.py       Superforecaster system prompt
  models/schemas.py     Pydantic request/response contracts
  services/rag_engine.py  Chunking, embeddings, retrieval, LLM call
frontend/
  src/App.jsx           Single-page layout
  src/components/       PredictorForm.jsx, ResultDisplay.jsx
```

## Backend Setup

Requires Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload
```

Interactive API docs: http://localhost:8000/docs

## Configuration

Create `backend/.env` (gitignored):

```
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Get a free key at https://console.groq.com.
Missing key → `/api/predict` returns HTTP 500 with instructions.

## Example Request

```powershell
$body = @{
  event_question    = "Will the S&P 500 close above 6000 by December 2026?"
  manual_crowd_prob = 0.5
  context_articles  = @("Paste article text here...")
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/api/predict -Method Post -Body $body -ContentType "application/json"
```

## Frontend Status

React source exists under `frontend/src`, but build scaffolding
(`package.json`, Vite config, entry point) is not committed yet — see roadmap below.

## Market Data Pipeline

A reproducible, leakage-safe preprocessing pipeline turns the raw Kaggle
global-markets dataset into model-ready S&P 500 data. The raw file is never modified.

### 1. Add the dataset

Place the downloaded Kaggle `.zip` (or extracted `.csv`) into:

```
data/raw/
```

The current source is `data/raw/global_financial_markets_2000_Now.csv`
(189,787 rows, 34 symbols, 2000-01-03 → 2026-08-20).

### 2. Run the pipeline

```powershell
.\.venv\Scripts\python.exe -m backend.pipeline.run_preprocess --with-targets
```

Options: `--symbol ^IXIC`, `--validation-start 2019-01-01`,
`--test-start 2023-01-01`, `--raw-path <file>`, `--processed-dir <dir>`.

### 3. Outputs (`data/processed/`)

| File | Contents |
|---|---|
| `sp500_clean.csv` | Cleaned, chronologically sorted OHLCV + metadata |
| `sp500_features.csv` | Clean data + engineered features |
| `sp500_targets.csv` | Forward-looking targets ONLY (`--with-targets`); never use as inputs |
| `sp500_features_train/validation/test.csv` | Configurable chronological splits (defaults: <2019 / 2019–2022 / ≥2023) |
| `preprocessing_report.json` | Full run report: cleaning summary, gaps, config, SHA-256 of raw input |

### 4. Feature definitions

All features at time *t* use only information available up to and including *t*
(warmup rows are empty until enough history exists):

| Column | Definition | First valid row |
|---|---|---|
| `return` | `close_t / close_(t-1) − 1` | 1 |
| `log_return` | `ln(close_t / close_(t-1))` | 1 |
| `ma_20 / ma_50 / ma_200` | Trailing mean of close (inclusive window) | 19 / 49 / 199 |
| `vol_20_annualized` | Sample std of last 20 log returns × √252 | 20 |
| `rolling_peak` | Running maximum of close from series start | 0 |
| `drawdown` | `close_t / rolling_peak − 1` | 0 |
| `daily_range` | `(high − low) / close` | 0 |
| `open_close_return` | `(close − open) / open` | 0 |

Targets: `target_return_1d/5d/10d` = forward close ratio; `target_future_vol_20`
= annualized std of the next 20 returns. Tail rows have empty targets.

### 5. Cleaning policy (conservative)

Nothing is fabricated: no interpolation, no price forward-fill. Rows are removed
only for malformed dates or missing/non-numeric OHLC prices (each removal is
logged). OHLC relationship violations and negative prices are **logged and kept**
(futures settlement artifacts; e.g., the April 2020 negative WTI print are real
history). Market-holiday gaps are left as-is. Rerunning produces byte-identical
outputs; `preprocessing_report.json` records the raw-file SHA-256.

### 6. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Covers filtering, sorting, dedupe, cleaning logs, exact feature math,
warmup handling, look-ahead-leakage prevention, split boundaries, and
byte-for-byte reproducibility.

## Roadmap

1. Frontend scaffolding (Vite + React + Tailwind)
2. S&P 500 market data ingestion and storage
3. Baseline trained forecasting model (walk-forward validation, no leakage)
4. Probability calibration and evaluation (Brier score, reliability curves)
5. Inference API serving the trained model
6. Dashboard and chat-style forecast UI
7. Prediction-market comparison (Kalshi) → automated information discrepancy
8. Tests, Docker, CI, deployment

## Security Notes

- Never commit `.env` or API keys; `.gitignore` already excludes them.
- The backend is unauthenticated by design for local development only — do not expose it publicly as-is.
