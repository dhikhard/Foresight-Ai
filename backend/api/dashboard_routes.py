"""FastAPI router for the ForeSight quantitative forecasting dashboard."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.services.quant_engine import QuantEngine

router = APIRouter(prefix="/api", tags=["dashboard"])
engine = QuantEngine()


class ShockSimulationRequest(BaseModel):
    portfolio_value: float = Field(default=100000.0, ge=1.0, description="Hypothetical portfolio equity in dollars")
    shock_pct: float = Field(default=-5.0, ge=-90.0, le=200.0, description="Market shock percentage (e.g. -5.0 for -5%)")


@router.get("/market/overview")
def get_market_overview(sample_points: int = Query(default=1500, ge=100, le=5000)):
    """Retrieve latest S&P 500 observation, indicators, and historical series."""
    try:
        data = engine.get_market_overview(chart_sample_points=sample_points)
        if data.get("status") == "unavailable":
            raise HTTPException(status_code=503, detail=data.get("message"))
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading market overview: {str(e)}")


@router.get("/models/forecast")
def get_model_forecast(feature_set: str = Query(default="v2", pattern="^(v1|v2)$")):
    """Compute out-of-sample forward forecast on latest available observation."""
    try:
        data = engine.get_forecast(feature_set=feature_set)
        if data.get("status") == "model_unavailable":
            raise HTTPException(status_code=503, detail=data.get("message"))
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing model forecast: {str(e)}")


@router.get("/models/performance")
def get_model_performance(feature_set: str = Query(default="v2", pattern="^(v1|v2)$")):
    """Retrieve model performance comparison across train, validation, and holdout test sets."""
    try:
        data = engine.get_performance_summary(feature_set=feature_set)
        if data.get("status") == "unavailable":
            raise HTTPException(status_code=503, detail=data.get("message"))
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving performance: {str(e)}")


@router.get("/risk/analysis")
def get_risk_analysis(portfolio_value: float = Query(default=100000.0, ge=1.0)):
    """Retrieve stress tests, historical drawdown, and Value-at-Risk calculations."""
    try:
        data = engine.get_risk_analysis(portfolio_value=portfolio_value)
        if data.get("status") == "unavailable":
            raise HTTPException(status_code=503, detail=data.get("message"))
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running risk analysis: {str(e)}")


@router.post("/risk/simulate")
def simulate_shock(request: ShockSimulationRequest):
    """Compute hypothetical portfolio impact under a custom market shock."""
    try:
        return engine.simulate_custom_shock(
            portfolio_value=request.portfolio_value,
            shock_pct=request.shock_pct
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error simulating shock: {str(e)}")
