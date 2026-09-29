"""Tests for retrieval. These run against whichever backend is available (embeddings or
TF-IDF fallback); both must satisfy the same correctness contract."""
from mf_assistant.retriever import get_retriever


def test_backend_available():
    r = get_retriever()
    assert r.name in {
        "hybrid(bm25+dense+rerank)", "hybrid(bm25+dense)", "embeddings+faiss", "tfidf"
    }


def test_expense_ratio_query_hits_ter_chunk():
    r = get_retriever()
    hits = r.retrieve("expense ratio of HDFC Flexi Cap Fund")
    assert hits, "expected at least one hit"
    top = hits[0]
    assert "flexi-cap" in top.source_url
    assert "expense" in top.topic.lower()


def test_lockin_query_hits_elss():
    r = get_retriever()
    hits = r.retrieve("lock-in period for ELSS tax saver")
    assert "elss" in hits[0].source_url.lower()


def test_scores_are_sorted_descending():
    r = get_retriever()
    hits = r.retrieve("minimum SIP for mid cap fund")
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_returns_at_most_top_k():
    r = get_retriever()
    hits = r.retrieve("benchmark of flexi cap", k=2)
    assert len(hits) <= 2
