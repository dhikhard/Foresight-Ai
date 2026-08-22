from typing import List

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    event_question: str = Field(..., min_length=1)
    manual_crowd_prob: float = Field(..., ge=0.0, le=1.0)
    context_articles: List[str] = Field(default_factory=list)


class Citation(BaseModel):
    quote: str


class PredictionResponse(BaseModel):
    event_question: str
    crowd_probability: float
    ai_probability: float
    information_edge: float
    reasoning: str
    citations: List[Citation]
