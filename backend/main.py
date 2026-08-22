from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.models.schemas import PredictionRequest, PredictionResponse
from backend.services.rag_engine import analyze_event


app = FastAPI(title="ForeSight API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    return analyze_event(request)
