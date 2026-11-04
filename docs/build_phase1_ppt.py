"""Build the ForeSightAI Phase 1 review deck (PPTX).

Reproducible generator for docs/ForeSightAI_Phase1.pptx.
All figures are the verified values recorded in docs/phase1_slides.md and
PHASE1_RESULTS.md; no numbers are invented here.
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "ForeSightAI_Phase1.pptx"

NAVY = RGBColor(0x14, 0x2A, 0x4C)
BLUE = RGBColor(0x1F, 0x5C, 0x99)
LIGHT = RGBColor(0xEA, 0xF1, 0xF8)
GREY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xA6, 0x1B, 0x1B)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

SW = prs.slide_width
SH = prs.slide_height


def _txbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    return tf


def _set_run(run, text, size, color, bold=False, italic=False):
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic


def title_bar(slide, title, subtitle=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SW, Inches(1.15))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tf = bar.text_frame
    tf.margin_left = Inches(0.5)
    tf.margin_top = Inches(0.12)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    _set_run(p.add_run(), title, 28, WHITE, bold=True)
    if subtitle:
        p2 = tf.add_paragraph()
        _set_run(p2.add_run(), subtitle, 13, LIGHT, italic=True)
    return bar


def bullets(slide, items, left=0.7, top=1.45, width=11.9, height=5.6, size=17):
    tf = _txbox(slide, left, top, width, height)
    first = True
    for item in items:
        text, level, *rest = item
        bold = rest[0] if rest else False
        color = rest[1] if len(rest) > 1 else RGBColor(0x20, 0x20, 0x20)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(6)
        marker = "•  " if level == 0 else "–  "
        if level == 0 and bold:
            marker = ""
        run = p.add_run()
        _set_run(run, marker + text, size if level == 0 else size - 2, color, bold=bold)
    return tf


def add_table(slide, headers, rows, left, top, width, height,
              col_widths=None, font=13, header_font=13, first_col_bold=True):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers),
                                   Inches(left), Inches(top), Inches(width), Inches(height))
    table = shape.table
    if col_widths:
        total = sum(col_widths)
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(width * w / total)
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = ""
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        _set_run(p.add_run(), h, header_font, WHITE, bold=True)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            _set_run(p.add_run(), str(val), font, RGBColor(0x20, 0x20, 0x20),
                     bold=(j == 0 and first_col_bold))
    return table


def footer(slide, idx, total, note=""):
    tf = _txbox(slide, 0.5, 7.02, 12.3, 0.4)
    p = tf.paragraphs[0]
    txt = f"ForeSightAI — Phase 1 review    |    Slide {idx}/{total}"
    if note:
        txt += f"    |    {note}"
    _set_run(p.add_run(), txt, 10, GREY)


# ---------------------------------------------------------------- slide 1
s = prs.slides.add_slide(BLANK)
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), SW, Inches(3.2))
band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background()
tf = _txbox(s, 0.8, 0.7, 11.7, 2.2)
p = tf.paragraphs[0]
_set_run(p.add_run(), "ForeSightAI", 46, WHITE, bold=True)
p2 = tf.add_paragraph()
_set_run(p2.add_run(), "AI-Driven Probabilistic Forecasting Engine", 22, LIGHT)
p3 = tf.add_paragraph()
_set_run(p3.add_run(), "Phase 1 — Progress, Results, and Next Steps", 18, WHITE, italic=True)

tf2 = _txbox(s, 0.8, 3.6, 11.7, 3.2)
rows = [
    ("Project team", "Chahel A (NNM23CC012) • Hardhik Taranath Shetty (NNM23CC017) • Poorvika (NNM23CC038)"),
    ("Department", "Computer and Communication Engineering"),
    ("Institution", "NMAM Institute of Technology"),
    ("Guide", "Prof. Chidananda T"),
    ("Review period", "Phase 1 (recorded evidence through 11 October 2026)"),
]
first = True
for k, v in rows:
    p = tf2.paragraphs[0] if first else tf2.add_paragraph()
    first = False
    _set_run(p.add_run(), f"{k}:  ", 15, BLUE, bold=True)
    _set_run(p.add_run(), v, 15, RGBColor(0x20, 0x20, 0x20))
    p.space_after = Pt(8)

# ---------------------------------------------------------------- slide 2
s = prs.slides.add_slide(BLANK)
title_bar(s, "Presentation Overview")
bullets(s, [
    ("Problem and objectives", 0, True),
    ("System architecture and data pipeline", 0, True),
    ("Feature engineering (V1 / V2) and leakage prevention", 0, True),
    ("Models implemented", 0, True),
    ("Results: accuracy and what they show", 0, True),
    ("Reproducibility and testing", 0, True),
    ("Progress to date", 0, True),
    ("Limitations, next steps, and conclusion", 0, True),
], size=20)

# ---------------------------------------------------------------- slide 3
s = prs.slides.add_slide(BLANK)
title_bar(s, "Problem Statement")
bullets(s, [
    ("Financial market returns are noisy and driven by many changing factors, which makes next-day forecasting hard.", 0),
    ("The project builds a systematic pipeline that processes historical market data, engineers features, trains regression models, and evaluates them against simple baselines.", 0),
    ("Central question: can machine learning models produce useful next-day S&P 500 return forecasts beyond simple baseline predictions?", 0, True, BLUE),
    ("Why baselines matter: a complex model is only useful if it improves on a simple alternative.", 0),
], size=18)

# ---------------------------------------------------------------- slide 4
s = prs.slides.add_slide(BLANK)
title_bar(s, "Objectives and Phase 1 Scope")
bullets(s, [
    ("Objectives", 0, True, NAVY),
    ("Process historical S&P 500 market data end to end", 1),
    ("Engineer features from historical price information", 1),
    ("Train and evaluate regression models", 1),
    ("Prevent data leakage during preprocessing and evaluation", 1),
    ("Compare learned models against simple baselines", 1),
    ("Produce reproducible reports and test results", 1),
    ("Phase 1 scope (implemented)", 0, True, NAVY),
    ("Data processing, feature engineering, baseline modeling, Ridge Regression, Elastic Net, evaluation, automated testing", 1),
    ("Not in Phase 1", 0, True, RED),
    ("External-data integration into the forecasting dataset, the dashboard, and the chatbot", 1),
], size=16)

# ---------------------------------------------------------------- slide 5 (architecture)
s = prs.slides.add_slide(BLANK)
title_bar(s, "System Architecture")
stages = [
    "Raw financial market data",
    "Data cleaning and preprocessing",
    "Feature engineering — V1 and V2",
    "Chronological train / validation / test split",
    "Training-only preprocessing",
    "Baseline models, Ridge, and Elastic Net",
    "Validation-based model selection",
    "Final evaluation and report generation",
]
left_x, right_x, box_w, box_h = 1.1, 7.1, 5.0, 0.62
ys = [1.55, 2.42, 3.29, 4.16]
for i, txt in enumerate(stages):
    col = 0 if i < 4 else 1
    row = i if i < 4 else i - 4
    x = left_x if col == 0 else right_x
    y = ys[row]
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(box_w), Inches(box_h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = BLUE if i % 2 == 0 else NAVY
    shp.line.fill.background()
    tfr = shp.text_frame
    tfr.margin_top = Inches(0.02)
    pp = tfr.paragraphs[0]
    pp.alignment = PP_ALIGN.CENTER
    _set_run(pp.add_run(), f"{i+1}. {txt}", 13, WHITE, bold=True)
for row in range(3):
    for col in (0, 1):
        x = (left_x if col == 0 else right_x) + box_w / 2 - 0.12
        ar = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(x),
                                Inches(ys[row] + box_h + 0.02),
                                Inches(0.24), Inches(0.18))
        ar.fill.solid(); ar.fill.fore_color.rgb = GREY; ar.line.fill.background()
ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(left_x + box_w + 0.15),
                        Inches(4.42), Inches(0.5), Inches(0.24))
ar.fill.solid(); ar.fill.fore_color.rgb = GREY; ar.line.fill.background()
tf = _txbox(s, 1.1, 5.15, 11.1, 1.5)
p = tf.paragraphs[0]
_set_run(p.add_run(), "The Phase 1 deliverable is the data and modeling pipeline. ", 14, RGBColor(0x20, 0x20, 0x20))
_set_run(p.add_run(), "External-data integration (isolated scaffolding, not integrated), the dashboard, and the chatbot (legacy prototype) are not part of the runnable Phase 1 verification.", 14, RGBColor(0x20, 0x20, 0x20), italic=True)

# ---------------------------------------------------------------- slide 6 (data)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Dataset and Preprocessing")
bullets(s, [
    ("Source: Kaggle \"Global Financial Markets\" dataset — ~189,787 rows, 34 symbols (2000-01-03 → 2026-08-20).", 0),
    ("Pipeline filters to ^GSPC (S&P 500): 6,698 daily rows retained.", 0),
    ("Conservative cleaning: deduplicate and sort chronologically; drop only malformed dates or missing/non-numeric OHLC.", 0),
    ("Nothing is fabricated — no interpolation, no price forward-fill; OHLC violations and negative prices are logged and kept.", 0),
    ("Raw file is read-only; SHA-256 recorded in preprocessing_report.json.", 0),
    ("Deterministic: re-running produces byte-identical outputs.", 0),
], size=17)

# ---------------------------------------------------------------- slide 7 (features)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Feature Engineering — V1 and V2")
bullets(s, [
    ("V1 — 10 backward-looking price/volatility features", 0, True, NAVY),
    ("return, log_return, ma_20, ma_50, ma_200, vol_20_annualized, rolling_peak, drawdown, daily_range, open_close_return", 1),
    ("V2 — 16 stationary features", 0, True, NAVY),
    ("ret_lag_1/2/5/10/20, mom_5/10/20, ma_dist_20/50/200, vol_60_annualized, vol_ratio_20_60, vol_change_5d, rsi_14, bollinger_pctb_20", 1),
    ("V2 replaces price-level features with stationary ones to reduce non-stationarity.", 0),
    ("All features at time t use only information available up to and including t; warm-up rows stay empty.", 0),
], size=15)

# ---------------------------------------------------------------- slide 8 (splits/leakage)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Chronological Splits and Leakage Prevention")
add_table(s,
          ["Split", "Boundary", "Modeling rows"],
          [["Train", "< 2019-01-01", "4,580"],
           ["Validation", "2019-01-01 ≤ d < 2023-01-01", "1,008"],
           ["Test", "≥ 2023-01-01 (final holdout)", "910"]],
          left=0.7, top=1.5, width=5.6, height=1.7, font=13, header_font=13)
bullets(s, [
    ("Train-only preprocessing: standardizer fit on TRAIN only, frozen for validation/test.", 0),
    ("Validation-only selection: Ridge alpha and Elastic Net (alpha, l1_ratio) chosen by VALIDATION MAE.", 0),
    ("Test scored once: after all decisions are frozen; never used for tuning.", 0),
    ("Rows missing any feature or the target are dropped before splitting — no imputation.", 0),
], left=6.6, top=1.5, width=6.1, height=4.5, size=15)

# ---------------------------------------------------------------- slide 9 (models)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Models Implemented")
add_table(s,
          ["Model", "Rule", "V1", "V2"],
          [["zero_return", "always predicts 0", "yes", "yes"],
           ["train_mean", "constant TRAIN-mean target", "yes", "yes"],
           ["persistence", "tomorrow = today's return", "yes", "no"],
           ["ridge", "closed-form ridge; alpha by validation MAE", "yes", "yes"],
           ["elastic_net", "NumPy coordinate descent; (alpha, l1_ratio) by validation MAE", "yes", "yes"]],
          left=0.7, top=1.5, width=11.9, height=3.4,
          col_widths=[1.6, 6.2, 0.8, 0.8], font=13)
bullets(s, [
    ("Canonical grids: alpha ∈ {0.01, 0.1, 1, 10, 100}; l1_ratio ∈ {0.1, 0.3, 0.5, 0.7, 0.9}.", 0),
    ("Elastic Net objective scaled by 1/(2n), so its alpha is not on the same numerical scale as Ridge's.", 0),
], top=5.1, size=14)

# ---------------------------------------------------------------- slide 10 (results V2 table)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Results — V2 Feature Set (16 features)", "Recorded run 2026-09-28T04:16:53Z · test split = final holdout")
add_table(s,
          ["Model", "Validation MAE", "Test MAE", "Test R²", "Test DirAcc"],
          [["zero_return", "0.009367", "0.006582", "−0.0074", "0.5604"],
           ["train_mean (baseline)", "0.009353", "0.006562", "−0.0041", "0.5604"],
           ["ridge (α=0.01)", "0.009377", "0.006642", "−0.0096", "0.5066"],
           ["elastic_net (α=0.01, l1=0.1)", "0.009353", "0.006562", "−0.0041", "0.5604"]],
          left=0.7, top=1.7, width=11.9, height=2.6, font=14, header_font=13)
bullets(s, [
    ("train_mean has the lowest test MAE and the least-negative R² — the regression models do not beat it.", 0, True, RED),
    ("Test R² is slightly negative for every model (predictions worse than the test-set mean).", 0),
    ("Elastic Net selects α=0.01, l1_ratio=0.1, shrinks all weights to zero ⇒ identical to train_mean.", 0),
], top=4.7, size=15)

# ---------------------------------------------------------------- slide 11 (results V1 table / accuracy)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Results — V1 Feature Set (10 features)", "Fresh demo run (canonical V1 report is stale / predates Elastic Net)")
add_table(s,
          ["Model", "Validation MAE", "Test MAE", "Test R²"],
          [["zero_return", "0.009367", "0.006582", "−0.0074"],
           ["train_mean (baseline)", "0.009353", "0.006562", "−0.0041"],
           ["persistence", "0.013913", "0.009602", "−1.0797"],
           ["ridge (α=0.01)", "0.009348", "0.006768", "−0.0626"],
           ["elastic_net (α=0.01, l1=0.1)", "0.009353", "0.006562", "−0.0041"]],
          left=0.7, top=1.7, width=11.9, height=3.2, font=14, header_font=13)
bullets(s, [
    ("V1 ridge has marginally lower validation MAE but clearly worse test MAE and R².", 0),
    ("persistence is clearly worse (test R² −1.08), confirming the evaluation discriminates between models.", 0),
    ("Directional accuracy (~0.56 constant, ~0.51 ridge) is not compared against a directional baseline ⇒ no directional skill claimed.", 0),
], top=5.1, size=15)

# ---------------------------------------------------------------- slide 12 (key findings)
s = prs.slides.add_slide(BLANK)
title_bar(s, "What the Results Show")
bullets(s, [
    ("No evaluated model demonstrates improved out-of-sample performance over the train-mean baseline.", 0, True, RED),
    ("Elastic Net is degenerate under the canonical grid — its apparent improvement is a constant-vs-zero effect, not learned signal.", 0),
    ("Ridge is at or below the baseline on the test period.", 0),
    ("Negative test R² arises because the test window has smaller target variance than train; predicting the train mean loses to the test-mean oracle.", 0),
    ("This is the honest expected outcome for near-unpredictable daily returns at this horizon — the value of Phase 1 is the reproducible, leakage-safe infrastructure, not predictive skill.", 0, True, BLUE),
    ("No claim of reliable stock-market prediction is supported by these measurements.", 0, True, RED),
], size=17)

# ---------------------------------------------------------------- slide 13 (repro & testing)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Reproducibility and Testing")
bullets(s, [
    ("One-command demo runner (run_demo.py): preprocess → V1 model → V2 features → V2 model.", 0),
    ("Writes only to a chosen output directory; refuses a non-empty directory without --force; emits demo_results.md.", 0),
    ("No network calls and no API keys required for the market pipeline.", 0),
    ("Test suite: 246 passed, 0 failed, 0 skipped (verified run, pytest 9.1.1).", 0, True, BLUE),
    ("Runtime verification: the demo generated V2 metrics numerically identical to the canonical report; canonical data/processed reports remained byte-unchanged.", 0),
    ("Deterministic pipeline: identical metrics across independent runs confirm reproducibility.", 0),
], size=17)

# ---------------------------------------------------------------- slide 14 (progress)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Progress to Date")
add_table(s,
          ["Week (approx.)", "Milestone", "Test count"],
          [["W8 (17–23 Aug)", "RAG prototype; preprocessing pipeline; Feature Set V2 + CLI", "≤ 37"],
           ["W9 (24–30 Aug)", "Ridge scale fix; V2 modeling wired; external-data scaffolding", "99 → 168"],
           ["W14 (28 Sep–4 Oct)", "Elastic Net; V2 report re-run; convergence hardening", "235 → 242"],
           ["W15 (5–11 Oct)", "Demo runner; docs (README, results, slides, log); path fix", "246"]],
          left=0.7, top=1.6, width=11.9, height=3.0,
          col_widths=[2.4, 7.4, 2.1], font=13, header_font=13)
bullets(s, [
    ("Documented test milestones: 99 → 165 → 168 → 235 → 242 → 246 passing.", 0),
    ("Weeks 1–7 and W10–13 have no repository evidence; they are marked \"needs confirmation\" in the progress log.", 0),
    ("All implementation, testing, and documentation work is in the working tree; nothing staged/committed since the last commit.", 0),
], top=4.9, size=15)

# ---------------------------------------------------------------- slide 15 (limitations)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Limitations (stated honestly)")
bullets(s, [
    ("No model beats the train-mean baseline out-of-sample; no reliable predictive skill is demonstrated.", 0),
    ("Elastic Net is degenerate on the canonical grid; the alpha-grid scale mismatch (1/(2n)) is unresolved and flagged.", 0),
    ("Single-asset scope (S&P 500) and a single regime window (2000–present).", 0),
    ("Single chronological split: the test period has already been observed once.", 0),
    ("Probabilistic calibration (Brier score, reliability curves) is not implemented, despite the project title.", 0),
    ("External data is scaffolded but not integrated; Kalshi needs credentials; Polymarket has a documented point-in-time caveat.", 0),
    ("The RAG chatbot and frontend are prototypes, not part of the runnable Phase 1 demo.", 0),
], size=16)

# ---------------------------------------------------------------- slide 16 (future work)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Next Steps / Future Work")
bullets(s, [
    ("Correct the Elastic Net alpha grid (an Elastic-Net-specific scale) and re-select using validation MAE only.", 0),
    ("Adopt walk-forward evaluation instead of a single split, with fresh holdout discipline.", 0),
    ("Add probability calibration and evaluation (Brier score, reliability curves) before any probabilistic claims.", 0),
    ("Integrate external data (news / prediction markets) with strict point-in-time safety; Kalshi needs credentials.", 0),
    ("Land the uncommitted modeling modules and tests as a dedicated commit; decide deliberately whether to commit context.txt.", 0),
    ("Build the dashboard only after the forecasting methodology is validated.", 0),
], size=17)

# ---------------------------------------------------------------- slide 17 (conclusion)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Conclusion")
bullets(s, [
    ("Phase 1 delivers a reproducible, leakage-safe S&P 500 forecasting and evaluation pipeline.", 0, True, NAVY),
    ("Deterministic preprocessing, V1/V2 features, chronological splits, train-only scaling, and validation-only selection.", 0),
    ("Baselines, Ridge, and Elastic Net are implemented in NumPy and covered by 246 passing tests.", 0),
    ("The honest finding: current models do not yet beat the train-mean baseline out-of-sample.", 0, True, RED),
    ("Further work is required to establish genuine predictive skill; the next steps above define that path.", 0),
], size=18)

# ---------------------------------------------------------------- slide 18 (appendix)
s = prs.slides.add_slide(BLANK)
title_bar(s, "Appendix — Report/Source Metadata Note")
bullets(s, [
    ("The committed reports in data/processed were generated from earlier run_modeling.py revisions and were intentionally preserved unchanged (not regenerated).", 0),
    ("modeling_report.json is an older V1 report (run 2026-08-27T06:05:24Z) and contains no elastic_net entry.", 0),
    ("modeling_report_v2.json records results generated at 2026-09-28T04:16:53Z.", 0),
    ("The current run_modeling.py emits a feature_set_comparison field, whereas the committed reports carry the older feature_set_winner_rule field.", 0),
    ("This is a metadata version mismatch only: the recorded numerical metrics were not independently re-run against the current source and are neither confirmed nor changed by it.", 0, True, BLUE),
], size=15)

prs.save(str(OUT))
print(f"Wrote {OUT}")
print(f"Slides: {len(prs.slides._sldIdLst)}")
