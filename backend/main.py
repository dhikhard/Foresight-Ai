from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.dashboard_routes import router as dashboard_router
from backend.models.schemas import PredictionRequest, PredictionResponse
from backend.services.rag_engine import analyze_event

app = FastAPI(
    title="ForeSight AI",
    description="Explainable Probabilistic Forecasting and Risk-Aware Decision Support",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Quantitative Dashboard API routes
app.include_router(dashboard_router)

# Static assets directory
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def read_root():
    """Serve the single-page quantitative finance dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "ok",
        "message": "ForeSight AI Backend Operational. Static UI not found.",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "ForeSight Quant & Forecasting Engine"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Legacy RAG probabilistic event forecast endpoint."""
    return analyze_event(request)
