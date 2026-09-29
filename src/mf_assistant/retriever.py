"""Retrieval layer (W3: RAG).

Two interchangeable backends behind one interface:
  * EmbeddingRetriever  — semantic search with sentence-transformers + FAISS (default)
  * TfidfRetriever      — lexical fallback (no heavy deps), used if embeddings unavailable

Both add a small lexical *topic boost*: chunks of the same scheme differ mainly by fact
type ("expense ratio" vs "benchmark" vs "lock-in"), so boosting exact topic-keyword hits
reliably disambiguates them.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

from . import config
from .models import RetrievedChunk


# --- corpus loading ------------------------------------------------------------
@lru_cache(maxsize=1)
def load_corpus() -> tuple:
    docs = json.loads(config.CORPUS_PATH.read_text(encoding="utf-8"))
    return tuple(docs)  # hashable for caching


def _topic_boost(query: str, topic: str) -> float:
    q_terms = {w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2}
    t_terms = set(re.findall(r"[a-z]+", topic.lower()))
    return config.TOPIC_BOOST * len(q_terms & t_terms)


def _to_chunk(doc: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=doc["id"], scheme=doc["scheme"], topic=doc["topic"], text=doc["text"],
        source_name=doc["source_name"], source_url=doc["source_url"],
        last_updated=doc.get("last_updated", config.SOURCES_LAST_UPDATED), score=score,
    )


# --- interface -----------------------------------------------------------------
class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def retrieve(self, query: str, k: int = config.TOP_K) -> List[RetrievedChunk]:
        ...


# --- semantic backend ----------------------------------------------------------
class EmbeddingRetriever(BaseRetriever):
    name = "embeddings+faiss"

    def __init__(self) -> None:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self._np = np
        self.docs = list(load_corpus())
        self.model = _get_model(config.EMBEDDING_MODEL)
        # Embed "scheme + topic + text" so scheme and fact-type both inform the vector.
        passages = [f"{d['scheme']} | {d['topic']} | {d['text']}" for d in self.docs]
        emb = self.model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype="float32")
        self.embeddings = emb
        self.index = _build_faiss(emb)

    def retrieve(self, query: str, k: int = config.TOP_K) -> List[RetrievedChunk]:
        np = self._np
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(q, dtype="float32")
        # Search wider, then re-rank with the topic boost and trim to k.
        n = min(len(self.docs), max(k * 2, 6))
        scores, idxs = self.index.search(q, n)
        ranked = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            boosted = float(score) + _topic_boost(query, self.docs[i]["topic"])
            ranked.append((boosted, float(score), i))
        ranked.sort(reverse=True)
        return [_to_chunk(self.docs[i], base) for _, base, i in ranked[:k]]


# --- lexical fallback ----------------------------------------------------------
class TfidfRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.docs = list(load_corpus())
        corpus = [f"{d['scheme']} {d['topic']} {d['text']}" for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = config.TOP_K) -> List[RetrievedChunk]:
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix).flatten()
        ranked = []
        for i, doc in enumerate(self.docs):
            boosted = float(sims[i]) + _topic_boost(query, doc["topic"])
            ranked.append((boosted, float(sims[i]), i))
        ranked.sort(reverse=True)
        return [_to_chunk(self.docs[i], base) for _, base, i in ranked[:k] if base > 0]


# --- helpers -------------------------------------------------------------------
@lru_cache(maxsize=2)
def _get_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def _build_faiss(embeddings):
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine
    index.add(embeddings)
    return index


@lru_cache(maxsize=1)
def get_retriever() -> BaseRetriever:
    """Return the best available retriever (embeddings preferred, TF-IDF fallback)."""
    try:
        return EmbeddingRetriever()
    except Exception:
        return TfidfRetriever()
