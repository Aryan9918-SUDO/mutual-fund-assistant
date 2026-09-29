"""Evaluation harness for the Facts-Only MF Assistant.

Runs a labelled query set through the pipeline and reports:
  * Routing accuracy    — did the query hit the right category (answer/advice/pii/...)?
  * Retrieval accuracy  — for factual queries, was the top cited source correct?
  * Refusal accuracy    — were advice/performance/PII queries correctly refused?
  * Grounded citation   — did every factual answer carry a source link?

Run:  python eval/run_eval.py

Uses the extractive fallback by default (no API cost); set GEMINI_API_KEY to eval with
Gemini phrasing. Exits non-zero if routing accuracy drops below THRESHOLD (used by CI).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mf_assistant import answer_query  # noqa: E402
from mf_assistant.models import AnswerKind  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "eval_set.json"
THRESHOLD = 0.90  # CI fails if routing accuracy falls below this

REFUSAL_KINDS = {"advice", "performance", "pii"}


def main() -> int:
    cases = json.loads(EVAL_SET.read_text(encoding="utf-8"))

    routing_ok = 0
    retrieval_total = retrieval_ok = 0
    refusal_total = refusal_ok = 0
    citation_total = citation_ok = 0
    failures = []

    for c in cases:
        ans = answer_query(c["query"])
        got = ans.kind.value
        expected = c["expect_kind"]

        # Routing
        if got == expected:
            routing_ok += 1
        else:
            failures.append(f"ROUTING  q={c['query']!r} expected={expected} got={got}")

        # Retrieval (only for factual answers with a labelled source)
        if expected == "answer":
            citation_total += 1
            if ans.source_url:
                citation_ok += 1
            if "expect_source" in c:
                retrieval_total += 1
                url = ans.source_url or ""
                if got == "answer" and c["expect_source"] in url:
                    retrieval_ok += 1
                else:
                    failures.append(
                        f"RETRIEVE q={c['query']!r} expected~{c['expect_source']} got={url}"
                    )

        # Refusal
        if expected in REFUSAL_KINDS:
            refusal_total += 1
            if got == expected:
                refusal_ok += 1

    n = len(cases)
    routing_acc = routing_ok / n
    metrics = {
        "total_cases": n,
        "routing_accuracy": round(routing_acc, 3),
        "retrieval_accuracy": round(retrieval_ok / retrieval_total, 3) if retrieval_total else None,
        "refusal_accuracy": round(refusal_ok / refusal_total, 3) if refusal_total else None,
        "grounded_citation_rate": round(citation_ok / citation_total, 3) if citation_total else None,
    }

    print("=" * 60)
    print("  Facts-Only MF Assistant — Evaluation Report")
    print("=" * 60)
    print(f"  Test cases            : {metrics['total_cases']}")
    print(f"  Routing accuracy      : {metrics['routing_accuracy']:.1%}")
    print(f"  Retrieval accuracy    : {metrics['retrieval_accuracy']:.1%}"
          f"  ({retrieval_ok}/{retrieval_total} factual queries)")
    print(f"  Refusal accuracy      : {metrics['refusal_accuracy']:.1%}"
          f"  ({refusal_ok}/{refusal_total} advice/PII/perf queries)")
    print(f"  Grounded citation rate: {metrics['grounded_citation_rate']:.1%}"
          f"  (every answer has a source link)")
    print("=" * 60)

    if failures:
        print("\n  Failures:")
        for f in failures:
            print("   -", f)

    (Path(__file__).resolve().parent / "last_report.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    return 0 if routing_acc >= THRESHOLD else 1


if __name__ == "__main__":
    raise SystemExit(main())
