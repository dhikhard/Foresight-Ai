"""Unit and Integration Tests for ForeSight Quantitative UI Engine & API."""

import pytest
from starlette.testclient import TestClient

from backend.main import app
from backend.services.quant_engine import QuantEngine


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def engine():
    return QuantEngine()


# ---------------------------------------------------------------------------
# QuantEngine Unit Tests
# ---------------------------------------------------------------------------

def test_market_overview(engine):
    overview = engine.get_market_overview(chart_sample_points=100)
    assert overview["status"] == "success"

    obs = overview["observation"]
    assert obs["date"] == "2026-08-20"
    assert obs["close"] > 7000.0
    assert obs["high"] >= obs["low"]
    assert isinstance(obs["daily_return"], float)

    ind = overview["indicators"]
    assert ind["vol_20_annualized"] > 0
    assert ind["vol_60_annualized"] > 0
    assert ind["current_drawdown"] <= 0.0  # drawdown is negative or zero
    assert ind["all_time_high"] >= obs["close"]

    features = overview["feature_breakdown"]
    assert len(features) == 16
    assert any(f["key"] == "rsi_14" for f in features)
    assert any(f["key"] == "bollinger_pctb_20" for f in features)

    chart = overview["chart_series"]
    assert len(chart) > 10
    assert chart[-1]["date"] == "2026-08-20"


def test_forecast_engine(engine):
    forecast = engine.get_forecast(feature_set="v2")
    assert forecast["status"] == "success"
    assert forecast["observation_date"] == "2026-08-20"
    assert forecast["target_horizon"] == "1 Trading Day Forward (t+1 close)"

    models = {m["id"]: m for m in forecast["models"]}
    assert "ridge" in models
    assert "train_mean" in models
    assert "elastic_net" in models
    assert "zero_return" in models

    # Check mathematical invariant: implied index = close * (1 + return)
    current_close = forecast["current_index_close"]
    for m in forecast["models"]:
        expected_price = current_close * (1.0 + m["predicted_return"])
        assert abs(m["implied_index_price"] - expected_price) < 1e-4

    # Verify elastic net degeneracy in canonical grid
    assert abs(models["elastic_net"]["predicted_return"] - models["train_mean"]["predicted_return"]) < 1e-6

    # Verify feature attributions
    attribs = forecast["ridge_feature_attributions"]
    assert len(attribs) == 16
    assert "feature" in attribs[0]
    assert "contribution" in attribs[0]


def test_performance_summary(engine):
    perf = engine.get_performance_summary(feature_set="v2")
    assert perf["status"] == "success"
    assert "models_comparison" in perf

    comp = {m["model_key"]: m for m in perf["models_comparison"]}
    assert "ridge" in comp
    assert "train_mean" in comp
    assert "elastic_net" in comp
    assert "zero_return" in comp

    # Train Mean should have test MAE matching PHASE1_RESULTS.md (0.006562)
    test_mae = comp["train_mean"]["test"]["mae"]
    assert round(test_mae, 6) == 0.006562

    # Ridge should have test MAE ~ 0.006642
    ridge_test_mae = comp["ridge"]["test"]["mae"]
    assert round(ridge_test_mae, 6) == 0.006642

    # Check holdout test evaluation series (910 observations)
    test_eval = perf["test_evaluation"]
    assert test_eval["available"] is True
    assert test_eval["test_sample_count"] == 910
    assert len(test_eval["series"]) == 910
    assert len(test_eval["residual_histogram"]) == 20


def test_risk_analysis(engine):
    portfolio = 250000.0
    risk = engine.get_risk_analysis(portfolio_value=portfolio)
    assert risk["status"] == "success"
    assert risk["portfolio_value"] == portfolio

    # Check prescribed shocks
    shocks = {s["shock_pct"]: s for s in risk["shocks"]}
    assert -10.0 in shocks
    assert -5.0 in shocks
    assert -3.0 in shocks
    assert 3.0 in shocks
    assert 5.0 in shocks

    # -5% shock on $250,000 should be -$12,500 PnL and $237,500 resulting equity
    assert shocks[-5.0]["portfolio_pnl"] == -12500.0
    assert shocks[-5.0]["resulting_portfolio_value"] == 237500.0

    # VaR checks
    rm = risk["risk_metrics"]
    assert rm["var_95_pct"] > 0
    assert rm["cvar_95_pct"] > rm["var_95_pct"]  # Expected shortfall >= VaR
    assert rm["var_99_pct"] > rm["var_95_pct"]   # 99% VaR > 95% VaR

    # Historical stress scenarios
    events = risk["historical_stress_events"]
    assert len(events) >= 4
    gfc = next(e for e in events if "Global Financial Crisis" in e["event"])
    assert gfc["peak_to_trough_drop_pct"] == -56.8
    assert abs(gfc["hypothetical_portfolio_pnl"] - (portfolio * -0.568)) < 1e-2


def test_simulate_custom_shock(engine):
    res = engine.simulate_custom_shock(portfolio_value=100000.0, shock_pct=-7.5)
    assert res["shock_pct"] == -7.5
    assert res["portfolio_pnl"] == -7500.0
    assert res["resulting_portfolio_value"] == 92500.0
    assert res["direction"] == "loss"


# ---------------------------------------------------------------------------
# FastAPI API Endpoints Integration Tests
# ---------------------------------------------------------------------------

def test_api_root_and_health(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "ForeSightAI" in res.text

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


def test_api_market_overview(client):
    res = client.get("/api/market/overview")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["observation"]["date"] == "2026-08-20"


def test_api_models_forecast(client):
    res = client.get("/api/models/forecast?feature_set=v2")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert len(data["models"]) >= 4

    # Test invalid feature set query
    bad_res = client.get("/api/models/forecast?feature_set=v99")
    assert bad_res.status_code == 422


def test_api_models_performance(client):
    res = client.get("/api/models/performance?feature_set=v2")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "models_comparison" in data


def test_api_risk_analysis(client):
    res = client.get("/api/risk/analysis?portfolio_value=150000")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["portfolio_value"] == 150000.0


def test_api_risk_simulate(client):
    payload = {"portfolio_value": 80000.0, "shock_pct": 4.5}
    res = client.post("/api/risk/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["portfolio_pnl"] == 3600.0
    assert data["resulting_portfolio_value"] == 83600.0
    assert data["direction"] == "gain"
