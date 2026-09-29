"""End-to-end pipeline tests: routing, grounding and privacy behaviour."""
from mf_assistant import answer_query
from mf_assistant.models import AnswerKind


def test_factual_answer_has_citation_and_is_short():
    a = answer_query("What is the expense ratio of HDFC Flexi Cap Fund?")
    assert a.kind == AnswerKind.ANSWER
    assert a.source_url and a.source_url.startswith("http")
    assert a.text.count(".") <= 4  # <= 3 sentences (allow trailing/decimal points)


def test_advice_is_refused():
    a = answer_query("Should I buy HDFC Flexi Cap Fund?")
    assert a.kind == AnswerKind.ADVICE
    assert a.is_refusal


def test_performance_is_refused():
    a = answer_query("What returns will HDFC Mid Cap give me?")
    assert a.kind == AnswerKind.PERFORMANCE


def test_pii_is_refused_and_not_echoed():
    pan = "ABCDE1234F"
    a = answer_query(f"My PAN is {pan}")
    assert a.kind == AnswerKind.PII
    assert pan not in a.text  # never echo the PII back


def test_out_of_scope():
    a = answer_query("what is the capital of France")
    assert a.kind == AnswerKind.OUT_OF_SCOPE


def test_empty_query():
    assert answer_query("   ").kind == AnswerKind.EMPTY


def test_semantic_paraphrase_routes_correctly():
    # No keyword overlap with "minimum SIP" -> relies on semantic retrieval / synonyms.
    a = answer_query("smallest amount to start a monthly plan in the mid cap fund")
    assert a.kind == AnswerKind.ANSWER
    assert "mid-cap" in a.source_url
