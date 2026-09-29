"""Lightweight, privacy-safe query logging (observability without storing PII).

We log the query *only* when it is not flagged as PII, plus which route fired and the top
retrieval score. Written as JSON Lines so it is trivial to analyse later.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config
from .models import Answer, AnswerKind


def log_query(query: str, answer: Answer) -> None:
    try:
        config.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": answer.kind.value,
            # Never persist the raw text of a PII-flagged query.
            "query": "[REDACTED_PII]" if answer.kind == AnswerKind.PII else query,
            "top_score": round(answer.confidence, 4) if answer.confidence is not None else None,
            "source_url": answer.source_url,
        }
        with config.LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never break the user-facing flow.
        pass
