"""Central configuration. All tunables live here so the rest of the code stays clean."""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]          # .../mutual-fund-assistant
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus.json"
SOURCES_CSV = DATA_DIR / "sources.csv"
INDEX_DIR = DATA_DIR / "index"                 # persisted FAISS index + embeddings
LOG_PATH = PROJECT_ROOT / "logs" / "queries.jsonl"

# --- Retrieval -----------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# "hybrid" (BM25 + dense + rerank) | "dense" (embeddings only) | "tfidf" (lexical only)
RETRIEVAL_MODE = os.environ.get("MF_RETRIEVAL_MODE", "hybrid")
USE_RERANKER = os.environ.get("MF_USE_RERANKER", "1") != "0"
TOP_K = 3                                       # chunks returned per query
CANDIDATE_K = 8                                # candidates before re-ranking
RRF_K = 60                                     # reciprocal-rank-fusion constant
MIN_SCORE = 0.25                               # below this => "not in my sources"
TOPIC_BOOST = 0.12                             # lexical boost per topic-keyword match

# --- Generation ----------------------------------------------------------------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
MAX_SENTENCES = 3                              # answers stay <= 3 sentences
GEN_TEMPERATURE = 0.1
GEN_MAX_TOKENS = 220

# --- Corpus metadata -----------------------------------------------------------
SOURCES_LAST_UPDATED = "2026-09-29"
AMC_NAME = "HDFC Mutual Fund"
EDU_LINK = "https://investor.sebi.gov.in/"     # used in safe refusals

DISCLAIMER = (
    "Facts-only. No investment advice. This assistant shares publicly available facts "
    "about mutual fund schemes from official AMC/SEBI/AMFI pages and does not recommend "
    "buying, selling or holding any scheme."
)

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Flexi Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver?",
    "How do I download my capital gains statement?",
]


def get_gemini_api_key() -> str | None:
    """Resolve the Gemini key from Streamlit secrets (if available) or the environment.

    Set MF_DISABLE_GEMINI=1 to force the deterministic extractive path (used in tests/eval).
    """
    if os.environ.get("MF_DISABLE_GEMINI") == "1":
        return None
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")
