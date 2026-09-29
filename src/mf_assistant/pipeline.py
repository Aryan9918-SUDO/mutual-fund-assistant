"""The orchestrator: routes a query through guardrails -> retrieval -> generation.

    query -> [PII? advice? performance?] -> [in-domain?] -> retrieve -> generate -> Answer

Keeping this as one small, readable function is deliberate: the routing *is* the product
logic (W1), and it should be obvious at a glance.
"""
from __future__ import annotations

from . import config, guardrails
from .generator import generate
from .logging_util import log_query
from .models import Answer, AnswerKind
from .retriever import get_retriever


class Pipeline:
    """Wraps a retriever so it can be constructed once and reused (e.g. cached in the UI)."""

    def __init__(self, retriever=None, enable_logging: bool = True) -> None:
        self.retriever = retriever or get_retriever()
        self.enable_logging = enable_logging

    def answer(self, query: str) -> Answer:
        ans = self._route(query)
        if self.enable_logging:
            log_query(query, ans)
        return ans

    def _route(self, query: str) -> Answer:
        q = (query or "").strip()
        if not q:
            return Answer(AnswerKind.EMPTY, "Please type a question.")

        # --- Guardrail 1: PII (highest priority; never store or echo it) ---
        pii = guardrails.detect_pii(q)
        if pii:
            return Answer(
                AnswerKind.PII,
                f"For your safety I can't accept or process personal information such as a "
                f"{pii}. Please remove it and ask only about scheme facts (e.g. expense "
                f"ratio, exit load, lock-in). No PII is stored.",
                source_name="SEBI Investor education", source_url=config.EDU_LINK,
            )

        # --- Guardrail 2: advice / opinion / portfolio ---
        if guardrails.is_advice(q):
            return Answer(
                AnswerKind.ADVICE,
                "I'm a facts-only assistant, so I can't give buy/sell or portfolio advice. "
                "I can share factual details like expense ratio, exit load, benchmark, "
                "riskometer or lock-in. For guidance, please consult a SEBI-registered adviser.",
                source_name="SEBI Investor education", source_url=config.EDU_LINK,
            )

        # --- Guardrail 3: performance / returns computation ---
        if guardrails.is_performance(q):
            return Answer(
                AnswerKind.PERFORMANCE,
                "I don't compute or compare returns or performance. Please refer to the "
                "scheme's official factsheet for past performance figures. I can share other "
                "facts like expense ratio, benchmark or minimum SIP.",
                source_name="HDFC Mutual Fund - Fund Factsheets",
                source_url="https://www.hdfcfund.com/",
            )

        # --- Retrieval + answerability ---
        chunks = self.retriever.retrieve(q)
        top_score = chunks[0].score if chunks else 0.0
        min_score = getattr(self.retriever, "min_score", config.MIN_SCORE)
        if not guardrails.in_domain(q) or not chunks or top_score < min_score:
            return Answer(
                AnswerKind.OUT_OF_SCOPE,
                "I don't have that fact in my official sources. I currently cover HDFC Large "
                "Cap, Flexi Cap, ELSS Tax Saver and Mid Cap funds, plus SEBI/AMFI concepts "
                "like expense ratio, riskometer and ELSS lock-in.",
                confidence=round(top_score, 4), retrieved=chunks,
            )

        top = chunks[0]
        text = generate(q, chunks)
        return Answer(
            AnswerKind.ANSWER, text,
            source_name=top.source_name, source_url=top.source_url,
            last_updated=top.last_updated, confidence=round(top_score, 4), retrieved=chunks,
        )


# Module-level convenience (lazily builds a shared pipeline).
_default_pipeline: Pipeline | None = None


def answer_query(query: str) -> Answer:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = Pipeline()
    return _default_pipeline.answer(query)
