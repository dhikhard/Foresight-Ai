import hashlib
import json
import math
import uuid
from typing import Any, Dict, List

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents
from fastapi import HTTPException
from langchain_groq import ChatGroq

from backend.core.config import get_settings
from backend.core.prompts import SUPERFORECASTER_SYSTEM_PROMPT
from backend.models.schemas import Citation, PredictionRequest, PredictionResponse


class HashEmbeddingFunction(EmbeddingFunction[Documents]):
    """Deterministic local embeddings for a no-external-embedding MVP."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def __call__(self, input: Documents) -> List[List[float]]:
        return [self._embed_text(text) for text in input]

    def _embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        tokens = [
            token.strip(".,!?;:()[]{}\"'").lower()
            for token in text.split()
            if token.strip(".,!?;:()[]{}\"'")
        ]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def _chunk_articles(articles: List[str], chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    chunks: List[str] = []

    for article in articles:
        cleaned = " ".join(article.split())
        if not cleaned:
            continue

        start = 0
        while start < len(cleaned):
            end = min(start + chunk_size, len(cleaned))
            chunks.append(cleaned[start:end])
            if end == len(cleaned):
                break
            start = max(0, end - overlap)

    return chunks


def _build_user_prompt(request: PredictionRequest, retrieved_context: List[str]) -> str:
    context_block = "\n\n---\n\n".join(retrieved_context) or "No context text was provided."
    return f"""
Event question:
{request.event_question}

Manual crowd probability:
{request.manual_crowd_prob}

Retrieved context:
{context_block}
""".strip()


def _parse_llm_json(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ValueError("LLM returned an unsupported response type.")
    return json.loads(content)


def analyze_event(request: PredictionRequest) -> PredictionResponse:
    settings = get_settings()
    if not settings.groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured. Add it to backend/.env or your environment.",
        )

    chunks = _chunk_articles(request.context_articles)
    collection_name = f"foresight_{uuid.uuid4().hex}"
    client = chromadb.Client()
    embedding_function = HashEmbeddingFunction()
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    try:
        if chunks:
            collection.add(
                ids=[f"chunk_{index}" for index in range(len(chunks))],
                documents=chunks,
            )
            results = collection.query(
                query_texts=[request.event_question],
                n_results=min(5, len(chunks)),
            )
            retrieved_context = results.get("documents", [[]])[0]
        else:
            retrieved_context = []

        llm = ChatGroq(
            model=settings.groq_model,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        response = llm.invoke(
            [
                ("system", SUPERFORECASTER_SYSTEM_PROMPT),
                ("human", _build_user_prompt(request, retrieved_context)),
            ]
        )
        parsed = _parse_llm_json(response.content)

        ai_probability = float(parsed.get("ai_probability"))
        ai_probability = max(0.0, min(1.0, ai_probability))
        citations = [
            Citation(quote=str(item.get("quote", "")).strip())
            for item in parsed.get("citations", [])
            if isinstance(item, dict) and str(item.get("quote", "")).strip()
        ]

        return PredictionResponse(
            event_question=request.event_question,
            crowd_probability=request.manual_crowd_prob,
            ai_probability=ai_probability,
            information_edge=ai_probability - request.manual_crowd_prob,
            reasoning=str(parsed.get("reasoning", "")).strip(),
            citations=citations,
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="Groq returned invalid JSON.") from exc
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
