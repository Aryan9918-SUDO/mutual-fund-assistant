"""Pydantic request/response schemas for the REST API (decoupled from internal models)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .models import Answer


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, examples=[
        "What is the expense ratio of HDFC Flexi Cap Fund?"
    ])


class ChunkOut(BaseModel):
    scheme: str
    topic: str
    text: str
    source_url: str
    score: float


class AnswerOut(BaseModel):
    kind: str = Field(..., description="answer | advice | performance | pii | out_of_scope | empty")
    text: str
    is_refusal: bool
    source_name: str | None = None
    source_url: str | None = None
    last_updated: str | None = None
    confidence: float | None = None
    retrieved: list[ChunkOut] = []

    @classmethod
    def from_answer(cls, a: Answer) -> AnswerOut:
        return cls(
            kind=a.kind.value,
            text=a.text,
            is_refusal=a.is_refusal,
            source_name=a.source_name,
            source_url=a.source_url,
            last_updated=a.last_updated,
            confidence=a.confidence,
            retrieved=[
                ChunkOut(scheme=c.scheme, topic=c.topic, text=c.text,
                         source_url=c.source_url, score=round(c.score, 4))
                for c in a.retrieved
            ],
        )


class HealthOut(BaseModel):
    status: str
    retriever: str
    corpus_chunks: int
    gemini_enabled: bool


class CompareOut(BaseModel):
    columns: list[str]
    schemes: dict
