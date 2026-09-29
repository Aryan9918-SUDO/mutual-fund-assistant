"""Typed data models shared across the package."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AnswerKind(str, Enum):
    """How the pipeline classified a query (W1: think-like-a-model routing)."""

    ANSWER = "answer"              # factual answer grounded in a source
    ADVICE = "advice"             # refused: opinion / buy-sell / portfolio
    PERFORMANCE = "performance"   # refused: returns / performance computation
    PII = "pii"                   # refused: contained personal data
    OUT_OF_SCOPE = "out_of_scope" # not covered by the corpus
    EMPTY = "empty"               # blank input


@dataclass
class RetrievedChunk:
    """One retrieved corpus chunk with its similarity score."""

    id: str
    scheme: str
    topic: str
    text: str
    source_name: str
    source_url: str
    last_updated: str
    score: float


@dataclass
class Answer:
    """The complete result returned to the UI / caller."""

    kind: AnswerKind
    text: str
    source_name: str | None = None
    source_url: str | None = None
    last_updated: str | None = None
    confidence: float | None = None
    retrieved: list[RetrievedChunk] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        return self.kind in {AnswerKind.ADVICE, AnswerKind.PERFORMANCE, AnswerKind.PII}
