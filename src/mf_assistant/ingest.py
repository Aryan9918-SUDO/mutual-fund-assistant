"""Ingestion pipeline: build and persist the vector index from the curated corpus.

Run:  python -m mf_assistant.ingest

This embeds every corpus chunk with the sentence-transformers model, builds a FAISS index,
and writes it (plus the raw embeddings and a metadata copy) to data/index/. The Streamlit
app builds its index in-memory at startup too, so persisting is optional — but it makes the
build step explicit, reproducible and inspectable, which is the point of an ETL stage.
"""
from __future__ import annotations

import json

from . import config
from .retriever import load_corpus


def build_index() -> dict:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    docs = list(load_corpus())
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    passages = [f"{d['scheme']} | {d['topic']} | {d['text']}" for d in docs]
    emb = np.asarray(
        model.encode(passages, normalize_embeddings=True, show_progress_bar=False),
        dtype="float32",
    )

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_DIR / "faiss.index"))
    np.save(config.INDEX_DIR / "embeddings.npy", emb)
    (config.INDEX_DIR / "meta.json").write_text(
        json.dumps(
            {
                "model": config.EMBEDDING_MODEL,
                "dim": int(emb.shape[1]),
                "num_chunks": len(docs),
                "chunk_ids": [d["id"] for d in docs],
                "last_updated": config.SOURCES_LAST_UPDATED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"chunks": len(docs), "dim": int(emb.shape[1]), "out": str(config.INDEX_DIR)}


if __name__ == "__main__":
    info = build_index()
    print(
        f"Built FAISS index: {info['chunks']} chunks, dim={info['dim']} "
        f"-> {info['out']}"
    )
