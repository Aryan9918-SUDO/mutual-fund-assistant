"""Facts-Only MF Assistant — a RAG FAQ system for HDFC mutual fund schemes.

Public API:
    from mf_assistant import answer_query, Answer, AnswerKind
"""
from .models import Answer, AnswerKind, RetrievedChunk
from .pipeline import Pipeline, answer_query

__all__ = ["answer_query", "Pipeline", "Answer", "AnswerKind", "RetrievedChunk"]
__version__ = "1.0.0"
