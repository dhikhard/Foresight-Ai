"""Feature Set V2 tests.

Every expected value is derived independently inside this file (inline
arithmetic, the statistics module, or hand-computed constants). The
implementation under test is never called to produce expectations.
Synthetic data only; the real dataset is never loaded.
"""

import math
import statistics

import pytest

from backend.pipeline.features_v2 import (
    FEATURE_COLUMNS_V2,
    FEATURE_WARMUP_V2,
    add_features_v2,
)


def make_rows(closes):
    """Cleaned-shape float rows (what add_features_v2 consumes)."""
    rows = []
    year, month, day = 2024, 1, 1
    for i, c in enumerate(closes):
        rows.append({
            "date": f"{year}-{month:02d}-{day:02d}",
            "open": c * 0.995, "high": c * 1.01, "low": c * 0.99,
            "close": float(c), "volume": 1000 + i, "symbol": "^GSPC",
            "asset_name": "S&P500", "asset_type": "Stock Index",
            "region": "Global",
        })
        day += 1
        if day > 28:
            day = 1
            month += 1
    return rows


# Deterministic pseudo-random positive series (no random module involved).
def wiggle_closes(n):
    return [
        round(100.0 * (1 + ((i * 37) % 19) / 95 - ((i * 23) % 13) / 85), 4)
        for i in range(n)
    ]


SMALL = [100, 101, 103, 102, 105, 107, 104, 108, 110, 109, 111, 113,
         112, 115, 114, 117, 116, 119, 118, 121, 120, 123, 122, 125, 124]


# ---------- contract ----------

def test_feature_contract_exact_16_columns_in_order():
    assert FEATURE_COLUMNS_V2 == (
        "ret_lag_1", "ret_lag_2", "ret_lag_5", "ret_lag_10", "ret_lag_20",
        "mom_5", "mom_10", "mom_20",
        "ma_dist_20", "ma_dist_50", "ma_dist_200",
        "vol_60_annualized", "vol_ratio_20_60", "vol_change_5d",
        "rsi_14", "bollinger_pctb_20",
    )


def test_empty_input_returns_empty_list():
    assert add_features_v2([]) == []


def test_output_preserves_input_and_adds_exactly_16_keys():
    rows = make_rows(SMALL)
    snapshot = [dict(r) for r in rows]
    out = add_features_v2(rows)
    assert len(out) == len(rows)
    for original, enriched in zip(snapshot, out):
        for key, value in original.items():
            assert enriched[key] == value          # inputs preserved
        new_keys = set(enriched) - set(original)
        assert new_keys == set(FEATURE_COLUMNS_V2)  # exactly 16 additions


# ---------- lagged returns ----------

def test_ret_lags_exact_against_manual_arithmetic():
    out = add_features_v2(make_rows(SMALL))
    # return[t] = SMALL[t]/SMALL[t-1] - 1; ret_lag_k[t] = return[t-k].
    assert out[1]["ret_lag_1"] is None                    # no prior-prior close
    assert out[2]["ret_lag_1"] == pytest.approx(101 / 100 - 1)
    assert out[3]["ret_lag_1"] == pytest.approx(103 / 101 - 1)
    # Cross-lag identity: ret_lag_2[t] must equal ret_lag_1[t-1].
    assert out[3]["ret_lag_2"] == pytest.approx(out[2]["ret_lag_1"])
    assert out[6]["ret_lag_5"] == pytest.approx(101 / 100 - 1)
    assert out[5]["ret_lag_5"] is None
    assert out[11]["ret_lag_10"] == pytest.approx(101 / 100 - 1)
    assert out[10]["ret_lag_10"] is None
    assert out[21]["ret_lag_20"] == pytest.approx(101 / 100 - 1)
    assert out[20]["ret_lag_20"] is None


# ---------- momentum ----------

def test_momentum_exact_against_direct_arithmetic():
    out = add_features_v2(make_rows(SMALL))
    assert out[4]["mom_5"] is None
    assert out[5]["mom_5"] == pytest.approx(SMALL[5] / SMALL[0] - 1)
    assert out[8]["mom_5"] == pytest.approx(SMALL[8] / SMALL[3] - 1)
    assert out[9]["mom_10"] is None
    assert out[10]["mom_10"] == pytest.approx(SMALL[10] / SMALL[0] - 1)
    assert out[19]["mom_20"] is None
    assert out[20]["mom_20"] == pytest.approx(SMALL[20] / SMALL[0] - 1)


# ---------- MA distance ----------

def test_ma_distance_exact_with_independent_trailing_means():
    out = add_features_v2(make_rows(SMALL))
    assert out[18]["ma_dist_20"] is None
    expected_first = SMALL[19] / (math.fsum(SMALL[0:20]) / 20) - 1
    assert out[19]["ma_dist_20"] == pytest.approx(expected_first)
    expected_shifted = SMALL[21] / (math.fsum(SMALL[2:22]) / 20) - 1
    assert out[21]["ma_dist_20"] == pytest.approx(expected_shifted)
    # 200-day distance can never mature within 25 rows.
    assert all(r["ma_dist_200"] is None for r in out)
    assert out[19]["ma_dist_20"] is not None  # anti-vacuity for the line above


# ---------- volatility structure ----------

def log_returns_of(closes):
    return [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]


def test_vol_60_matches_statistics_stdev_times_sqrt252():
    closes = wiggle_closes(80)
    out = add_features_v2(make_rows(closes))
    lrs = log_returns_of(closes)
    expected = statistics.stdev(lrs[10:70]) * math.sqrt(252)   # window ends at t=70
    assert out[70]["vol_60_annualized"] == pytest.approx(expected, rel=1e-9)


def test_vol_ratio_exact_and_warmup_boundary():
    closes = wiggle_closes(80)
    out = add_features_v2(make_rows(closes))
    lrs = log_returns_of(closes)
    s20 = statistics.stdev(lrs[50:70])   # rows 51..70 (helper pos p -> row p+1)
    s60 = statistics.stdev(lrs[10:70])   # rows 11..70
    # Annualization cancels in the ratio: sqrt(252)*s20 / (sqrt(252)*s60).
    assert out[59]["vol_ratio_20_60"] is None
    assert out[70]["vol_ratio_20_60"] == pytest.approx(s20 / s60, rel=1e-9)


def test_vol_change_5d_exact_and_warmup_boundary():
    closes = wiggle_closes(80)
    out = add_features_v2(make_rows(closes))
    lrs = log_returns_of(closes)
    s_now = statistics.stdev(lrs[20:40])     # vol20 over rows 21..40 (t=40)
    s_before = statistics.stdev(lrs[15:35])  # vol20 over rows 16..35 (t=35)
    assert out[24]["vol_change_5d"] is None
    assert out[40]["vol_change_5d"] == pytest.approx(s_now / s_before - 1, rel=1e-9)


def test_zero_volatility_series_yields_none_for_ratio_change_pctb():
    out = add_features_v2(make_rows([100.0] * 70))
    t = 65
    assert out[t]["vol_ratio_20_60"] is None       # denominator exactly zero
    assert out[t]["vol_change_5d"] is None
    assert out[t]["bollinger_pctb_20"] is None     # bands collapsed (4*sigma=0)
    # Deterministic degenerate values elsewhere:
    assert out[t]["ret_lag_1"] == 0.0
    assert out[t]["mom_20"] == 0.0
    assert out[t]["ma_dist_20"] == 0.0
    assert out[t]["rsi_14"] == 50.0                # documented both-zero rule


# ---------- RSI (Wilder convention, hand-computed) ----------

def test_rsi_seed_on_alternating_series_is_exactly_50():
    closes = [100, 101, 100, 101, 100, 101, 100, 101, 100, 101,
              100, 101, 100, 101, 100, 101, 103]
    out = add_features_v2(make_rows(closes))
    assert out[13]["rsi_14"] is None               # seed incomplete
    # First 14 changes: seven +1 gains, seven -1 losses -> seed averages 0.5/0.5
    # -> RS = 1 -> RSI = 100 - 100/2 = 50 exactly.
    assert out[14]["rsi_14"] == 50.0
    # Update at i=15 is the next alternating change (+1): 101 - 100.
    g1 = (0.5 * 13 + 1) / 14
    l1 = (0.5 * 13 + 0) / 14
    assert out[15]["rsi_14"] == pytest.approx(100.0 - 100.0 / (1.0 + g1 / l1))
    # Update at i=16 is the engineered +2 jump: closes[16]=103, closes[15]=101.
    g2 = (g1 * 13 + 2) / 14
    l2 = (l1 * 13 + 0) / 14
    assert out[16]["rsi_14"] == pytest.approx(100.0 - 100.0 / (1.0 + g2 / l2))


def test_rsi_extremes_and_flat_rule():
    rising = add_features_v2(make_rows([100.0 + i for i in range(20)]))
    assert rising[14]["rsi_14"] == 100.0           # zero-loss rule
    falling = add_features_v2(make_rows([300.0 - i for i in range(20)]))
    assert falling[14]["rsi_14"] == 0.0            # zero-gain rule
    flat = add_features_v2(make_rows([100.0] * 16))
    assert flat[14]["rsi_14"] == 50.0              # both-zero rule


def test_rsi_matches_independent_exponential_weighting_reference():
    closes = wiggle_closes(90)
    out = add_features_v2(make_rows(closes))
    # Independent formulation: Wilder smoothing as exponential weighting with
    # alpha = 1/14 over gains/losses (seeded by the plain mean of the first 14).
    alpha = 1.0 / 14.0
    diffs = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_g = sum(max(d, 0.0) for d in diffs[:14]) / 14
    avg_l = sum(max(-d, 0.0) for d in diffs[:14]) / 14
    reference = {13: None}
    rsi = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    reference[14] = rsi                            # index in closes-space
    for k, d in enumerate(diffs[14:], start=15):
        avg_g = (1 - alpha) * avg_g + alpha * max(d, 0.0)
        avg_l = (1 - alpha) * avg_l + alpha * max(-d, 0.0)
        reference[k] = 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    for t in range(14, 60):
        assert out[t]["rsi_14"] == pytest.approx(reference[t], rel=1e-9), f"t={t}"


# ---------- Bollinger %b ----------

def test_bollinger_pctb_exact_with_statistics_module():
    closes = wiggle_closes(60)
    out = add_features_v2(make_rows(closes))
    assert out[18]["bollinger_pctb_20"] is None
    window = closes[7:27]                           # ends at t=26
    middle = math.fsum(window) / 20
    sigma = statistics.stdev(window)
    expected = (closes[26] - (middle - 2 * sigma)) / ((middle + 2 * sigma) - (middle - 2 * sigma))
    assert out[26]["bollinger_pctb_20"] == pytest.approx(expected, rel=1e-9)


# ---------- warm-up table ----------

EXPECTED_WARMUP = {
    "ret_lag_1": 2, "ret_lag_2": 3, "ret_lag_5": 6, "ret_lag_10": 11,
    "ret_lag_20": 21,
    "mom_5": 5, "mom_10": 10, "mom_20": 20,
    "ma_dist_20": 19, "ma_dist_50": 49, "ma_dist_200": 199,
    "vol_60_annualized": 60, "vol_ratio_20_60": 60, "vol_change_5d": 25,
    "rsi_14": 14, "bollinger_pctb_20": 19,
}


def test_module_warmup_constants_match_documented_table():
    assert FEATURE_WARMUP_V2 == EXPECTED_WARMUP


def test_each_feature_none_until_its_own_lookback_then_real():
    out = add_features_v2(make_rows(wiggle_closes(230)))
    for name, warmup in EXPECTED_WARMUP.items():
        for t in range(warmup):
            assert out[t][name] is None, f"{name} defined early at t={t}"
        first = out[warmup][name]
        assert isinstance(first, float) and math.isfinite(first), \
            f"{name} not real-valued at its warmup index {warmup}"


# ---------- look-ahead safety ----------

@pytest.mark.parametrize("probe", [25, 60, 150, 199])
def test_future_perturbation_leaves_past_features_bit_identical(probe):
    closes = wiggle_closes(230)
    base = add_features_v2(make_rows(closes))

    perturbed_closes = [c * 1.07 if i > probe else c for i, c in enumerate(closes)]
    perturbed = add_features_v2(make_rows(perturbed_closes))

    mature_guard = {
        25: ["ret_lag_5", "mom_20", "ma_dist_20", "vol_change_5d",
             "rsi_14", "bollinger_pctb_20"],
        60: ["ret_lag_20", "mom_20", "ma_dist_50", "vol_60_annualized",
             "vol_ratio_20_60", "rsi_14"],
        150: ["ret_lag_20", "mom_20", "ma_dist_50", "vol_60_annualized",
              "vol_ratio_20_60", "rsi_14"],
        199: ["ret_lag_20", "mom_20", "ma_dist_200", "vol_60_annualized",
              "vol_ratio_20_60", "bollinger_pctb_20"],
    }[probe]
    for name in mature_guard:
        assert isinstance(base[probe][name], float), f"guard failed for {name}"

    for name in FEATURE_COLUMNS_V2:
        expected = base[probe][name]
        actual = perturbed[probe][name]
        if expected is None:
            assert actual is None, f"leakage: {name} appeared at {probe}"
        else:
            # Exact equality: features depend only on the unchanged prefix,
            # so recomputation must reproduce identical IEEE-754 doubles.
            assert actual == expected, f"leakage in {name} at {probe}"


# ---------- determinism / purity ----------

def test_determinism_same_input_same_output():
    rows = make_rows(wiggle_closes(120))
    first = add_features_v2(rows)
    second = add_features_v2(rows)
    assert first == second
    # Determinism must not be vacuous: the run must have produced real
    # feature content, and different inputs must produce different outputs.
    assert isinstance(first[100]["rsi_14"], float)
    other = add_features_v2(make_rows(wiggle_closes(120)[::-1]))
    assert [r["mom_5"] for r in other[:30]] != [r["mom_5"] for r in first[:30]]


# ---------- contamination ----------

def test_no_target_contamination_anywhere():
    out = add_features_v2(make_rows(wiggle_closes(220)))
    forbidden = {"target_return_1d", "target_return_5d", "target_return_10d",
                 "target_future_vol_20"}
    assert not (set(out[0].keys()) & forbidden)
    assert not any(c.startswith("target_") for c in FEATURE_COLUMNS_V2)


# ---------- malformed / non-positive inputs ----------

def test_non_positive_close_degrades_gracefully_per_v1_conventions():
    closes = wiggle_closes(80)
    closes[40] = 0.0                                # corrupt one observation
    out = add_features_v2(make_rows(closes))

    # Returns touching the corrupted pair are undefined; ret_lag_k[t] reads
    # return[t-k], so the damage surfaces at t=41 and t=42, not t=40.
    assert out[40]["ret_lag_1"] is not None         # return[39] is a healthy pair
    assert out[41]["ret_lag_1"] is None             # return[40] spans close[39]->0
    assert out[42]["ret_lag_1"] is None             # return[41] spans 0->close[41]
    assert out[45]["mom_5"] is None                 # window includes close[40]
    for t in range(40, 60):
        assert out[t]["ma_dist_20"] is None         # every 20-window contains t=40
    assert out[60]["ma_dist_20"] is not None        # recovered: window starts at 41
    for t in range(40, 80):
        assert out[t]["vol_60_annualized"] is None  # every 60-window spans t=40
    for t in range(40, 80):
        assert out[t]["rsi_14"] is None             # dead-series policy from t=40
    assert out[39]["rsi_14"] is not None            # ...but healthy up to t=39
    for t in range(60, 80):
        assert out[t]["bollinger_pctb_20"] is not None  # recovered once window passes
