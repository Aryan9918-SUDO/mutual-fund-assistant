"""Unit tests for the safety guardrails (pure functions -> easy to test exhaustively)."""
import pytest

from mf_assistant import guardrails as g


@pytest.mark.parametrize(
    "text,expected",
    [
        ("My PAN is ABCDE1234F", "PAN"),
        ("aadhaar 1234 5678 9012", "Aadhaar"),
        ("call me on 9876543210", "phone number"),
        ("email test@example.com", "email address"),
        ("my otp is 445566", "OTP"),
        ("folio number is 123456789", "account number"),
    ],
)
def test_detect_pii_positive(text, expected):
    assert g.detect_pii(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "What is the expense ratio of HDFC Flexi Cap Fund?",
        "lock-in period for ELSS",
        "benchmark of the large cap fund",
    ],
)
def test_detect_pii_negative(text):
    assert g.detect_pii(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Should I buy HDFC Flexi Cap?",
        "Which is the best HDFC fund?",
        "Is HDFC Mid Cap Fund good for me?",
        "Can you recommend a tax saver?",
        "should i sell my units",
    ],
)
def test_is_advice_positive(text):
    assert g.is_advice(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "What is the expense ratio?",
        "How do I download my statement?",
        "What is the benchmark?",
    ],
)
def test_is_advice_negative(text):
    assert g.is_advice(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "What returns will I get?",
        "CAGR of HDFC Flexi Cap",
        "compare returns of two funds",
        "how much will I earn in 5 years",
    ],
)
def test_is_performance_positive(text):
    assert g.is_performance(text) is True


def test_in_domain():
    assert g.in_domain("expense ratio of flexi cap") is True
    assert g.in_domain("what is the capital of France") is False
    assert g.in_domain("tell me a joke") is False
