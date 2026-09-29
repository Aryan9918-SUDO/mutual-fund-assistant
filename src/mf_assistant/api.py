"""FastAPI backend for the Facts-Only MF Assistant.

Exposes the same guardrails + RAG pipeline as a REST service, so any client (the Streamlit
UI, a script, another service) can consume it. Interactive docs at /docs.

Run:  uvicorn mf_assistant.api:app --reload      (or: make api)
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from . import config
from .pipeline import Pipeline
from .schemas import AnswerOut, AskRequest, CompareOut, HealthOut

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the pipeline (loads models + index) once at startup.
    _state["pipeline"] = Pipeline()
    _state["facts"] = json.loads(
        (config.DATA_DIR / "scheme_facts.json").read_text(encoding="utf-8")
    )
    yield
    _state.clear()


app = FastAPI(
    title="Facts-Only MF Assistant API",
    version="1.0.0",
    description="RAG FAQ service for HDFC mutual fund schemes. Facts-only, cited, no advice.",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    pipe: Pipeline = _state["pipeline"]
    return HealthOut(
        status="ok",
        retriever=pipe.retriever.name,
        corpus_chunks=len(pipe.retriever.docs),
        gemini_enabled=bool(config.get_gemini_api_key()),
    )


@app.post("/ask", response_model=AnswerOut, tags=["qa"])
def ask(req: AskRequest) -> AnswerOut:
    """Answer a factual question (or refuse advice/performance/PII), grounded in one source."""
    ans = _state["pipeline"].answer(req.query)
    return AnswerOut.from_answer(ans)


@app.get("/compare", response_model=CompareOut, tags=["qa"])
def compare() -> CompareOut:
    """Return the side-by-side factual comparison table for all covered schemes."""
    facts = _state["facts"]
    return CompareOut(columns=facts["columns"], schemes=facts["schemes"])
