"""Retrieval layer (W3: RAG).

Backends behind one interface:
  * HybridRetriever    — BM25 (lexical) + dense (embeddings), fused with Reciprocal Rank
                         Fusion, then re-ranked by a cross-encoder (default).
  * EmbeddingRetriever — dense-only semantic search (sentence-transformers + FAISS).
  * TfidfRetriever     — pure lexical fallback (no heavy deps).

`get_retriever()` picks the best available backend and degrades gracefully. All backends
expose `.score` in ~[0,1] (higher = more relevant) so the pipeline threshold is uniform.
A small lexical *topic boost* disambiguates same-scheme chunks (expense ratio vs benchmark).
"""
from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from . import config
from .models import RetrievedChunk


# --- corpus loading ------------------------------------------------------------
@lru_cache(maxsize=1)
def load_corpus() -> tuple:
    docs = json.loads(config.CORPUS_PATH.read_text(encoding="utf-8"))
    return tuple(docs)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _topic_boost(query: str, topic: str) -> float:
    q_terms = {w for w in _tokenize(query) if len(w) > 2}
    t_terms = set(_tokenize(topic))
    return config.TOPIC_BOOST * len(q_terms & t_terms)


def _to_chunk(doc: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=doc["id"], scheme=doc["scheme"], topic=doc["topic"], text=doc["text"],
        source_name=doc["source_name"], source_url=doc["source_url"],
        last_updated=doc.get("last_updated", config.SOURCES_LAST_UPDATED), score=score,
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# --- interface -----------------------------------------------------------------
class BaseRetriever(ABC):
    name: str = "base"
    docs: list[dict] = []
    # Minimum top score to treat a result as in-corpus. Cross-encoder scores live on a
    # different scale than cosine, so each backend declares its own threshold.
    min_score: float = config.MIN_SCORE

    @abstractmethod
    def retrieve(self, query: str, k: int = config.TOP_K) -> list[RetrievedChunk]:
        ...


# --- dense (embeddings + FAISS) ------------------------------------------------
class EmbeddingRetriever(BaseRetriever):
    name = "embeddings+faiss"

    def __init__(self) -> None:
        import numpy as np

        self._np = np
        self.docs = list(load_corpus())
        self.model = _get_model(config.EMBEDDING_MODEL)
        passages = [f"{d['scheme']} | {d['topic']} | {d['text']}" for d in self.docs]
        emb = np.asarray(
            self.model.encode(passages, normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        self.embeddings = emb
        self.index = _build_faiss(emb)

    def _cosine_scores(self, query: str):
        np = self._np
        q = np.asarray(
            self.model.encode([query], normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )
        return (self.embeddings @ q[0])  # normalized vectors -> cosine

    def retrieve(self, query: str, k: int = config.TOP_K) -> list[RetrievedChunk]:
        sims = self._cosine_scores(query)
        ranked = sorted(
            ((float(sims[i]) + _topic_boost(query, d["topic"]), float(sims[i]), i)
             for i, d in enumerate(self.docs)),
            reverse=True,
        )
        return [_to_chunk(self.docs[i], base) for _, base, i in ranked[:k]]


# --- hybrid (BM25 + dense + RRF + cross-encoder rerank) ------------------------
class HybridRetriever(BaseRetriever):
    name = "hybrid(bm25+dense+rerank)"

    def __init__(self) -> None:
        from rank_bm25 import BM25Okapi

        self.dense = EmbeddingRetriever()
        self.docs = self.dense.docs
        corpus_tokens = [_tokenize(f"{d['scheme']} {d['topic']} {d['text']}") for d in self.docs]
        self.bm25 = BM25Okapi(corpus_tokens)
        self.reranker = _get_cross_encoder() if config.USE_RERANKER else None
        if self.reranker is None:
            self.name = "hybrid(bm25+dense)"
        else:
            # Cross-encoder sigmoid: out-of-scope ~0.0, in-scope >=~0.14 -> low threshold.
            self.min_score = 0.05

    def _rrf_candidates(self, query: str, n: int) -> list[int]:
        """Fuse dense and BM25 rankings via Reciprocal Rank Fusion; return candidate idxs."""
        dense_scores = self.dense._cosine_scores(query)
        dense_rank = sorted(range(len(self.docs)), key=lambda i: -float(dense_scores[i]))
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_rank = sorted(range(len(self.docs)), key=lambda i: -float(bm25_scores[i]))

        rrf: dict[int, float] = {}
        for rank_list in (dense_rank[:n], bm25_rank[:n]):
            for rank, idx in enumerate(rank_list):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (config.RRF_K + rank)
        return sorted(rrf, key=lambda i: -rrf[i])[:n]

    def retrieve(self, query: str, k: int = config.TOP_K) -> list[RetrievedChunk]:
        cand = self._rrf_candidates(query, config.CANDIDATE_K)
        if not cand:
            return []

        if self.reranker is not None:
            pairs = [
                (query, f"{self.docs[i]['scheme']} ({self.docs[i]['topic']}). "
                        f"{self.docs[i]['text']}")
                for i in cand
            ]
            logits = self.reranker.predict(pairs)
            scored = [
                (_sigmoid(float(logits[j])) + _topic_boost(query, self.docs[i]["topic"]),
                 _sigmoid(float(logits[j])), i)
                for j, i in enumerate(cand)
            ]
        else:
            # No reranker: score candidates by dense cosine (calibrated ~[0,1]) + boost.
            sims = self.dense._cosine_scores(query)
            scored = [
                (float(sims[i]) + _topic_boost(query, self.docs[i]["topic"]), float(sims[i]), i)
                for i in cand
            ]

        scored.sort(reverse=True)
        return [_to_chunk(self.docs[i], base) for _, base, i in scored[:k]]


# --- lexical fallback ----------------------------------------------------------
class TfidfRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.docs = list(load_corpus())
        corpus = [f"{d['scheme']} {d['topic']} {d['text']}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = config.TOP_K) -> list[RetrievedChunk]:
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix).flatten()
        ranked = sorted(
            ((float(sims[i]) + _topic_boost(query, d["topic"]), float(sims[i]), i)
             for i, d in enumerate(self.docs)),
            reverse=True,
        )
        return [_to_chunk(self.docs[i], base) for _, base, i in ranked[:k] if base > 0]


# --- model loaders (cached singletons) -----------------------------------------
@lru_cache(maxsize=2)
def _get_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


@lru_cache(maxsize=1)
def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.CROSS_ENCODER_MODEL)


def _build_faiss(embeddings):
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine on normalized vectors
    index.add(embeddings)
    return index


@lru_cache(maxsize=1)
def get_retriever() -> BaseRetriever:
    """Return the best available retriever for the configured mode, degrading gracefully."""
    mode = config.RETRIEVAL_MODE
    modes: dict[str, list[type[BaseRetriever]]] = {
        "hybrid": [HybridRetriever, EmbeddingRetriever, TfidfRetriever],
        "dense": [EmbeddingRetriever, TfidfRetriever],
        "tfidf": [TfidfRetriever],
    }
    order = modes.get(mode, [HybridRetriever, EmbeddingRetriever, TfidfRetriever])

    for cls in order:
        try:
            return cls()
        except Exception:
            continue
    return TfidfRetriever()
