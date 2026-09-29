"""Tests for the FastAPI backend using Starlette's TestClient."""
import pytest
from fastapi.testclient import TestClient

from mf_assistant.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` triggers startup/shutdown (lifespan)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["corpus_chunks"] > 0
    assert "retriever" in body


def test_ask_factual(client):
    r = client.post("/ask", json={"query": "What is the expense ratio of HDFC Flexi Cap Fund?"})
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "answer"
    assert body["source_url"].startswith("http")
    assert body["is_refusal"] is False
    assert len(body["retrieved"]) >= 1


def test_ask_advice_refused(client):
    r = client.post("/ask", json={"query": "Should I buy HDFC Flexi Cap Fund?"})
    assert r.json()["kind"] == "advice"
    assert r.json()["is_refusal"] is True


def test_ask_pii_not_echoed(client):
    r = client.post("/ask", json={"query": "My PAN is ABCDE1234F"})
    body = r.json()
    assert body["kind"] == "pii"
    assert "ABCDE1234F" not in body["text"]


def test_ask_validation_error(client):
    r = client.post("/ask", json={"query": ""})
    assert r.status_code == 422  # fails min_length validation


def test_compare(client):
    r = client.get("/compare")
    assert r.status_code == 200
    body = r.json()
    assert "columns" in body and "schemes" in body
    assert len(body["schemes"]) >= 3
