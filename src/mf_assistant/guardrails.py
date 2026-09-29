"""Safety guardrails (W1: decide answer vs. refuse).

Three refusal categories plus a domain gate. Each detector is a pure function so it is
trivially unit-testable.
"""
from __future__ import annotations

import re

# 1) PII we must never accept or store. -----------------------------------------
PII_PATTERNS = {
    "PAN": r"\b[A-Za-z]{5}[0-9]{4}[A-Za-z]\b",
    "Aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "phone number": r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b",
    "email address": r"\b[\w.\-]+@[\w\-]+\.[A-Za-z]{2,}\b",
    "OTP": r"\b(?:otp|one[\-\s]?time\s?password)\b[\s:is]*\d{4,8}\b",
    "account number": r"\b(?:a/c|acc(?:ount)?|folio)\b\D{0,15}\d{6,}\b",
}

# 2) Opinion / advice / portfolio questions we must refuse. ----------------------
ADVICE_PATTERNS = [
    r"\bshould i\b", r"\bshould we\b", r"\bis it (?:good|bad|worth|safe|better)\b",
    r"\bwhich (?:is )?(?:the )?best\b", r"\bbest (?:fund|scheme|option)\b",
    r"\brecommend\b", r"\bsuggest\b", r"\badvice\b", r"\badvise\b",
    r"\bbuy or sell\b", r"\bshould i buy\b", r"\bshould i sell\b", r"\binvest in\b",
    r"\bworth (?:buying|investing|it)\b", r"\bgood (?:to|for) invest\b",
    r"\bgood for me\b", r"\bright for me\b", r"\bsuitable for me\b",
    r"\bwill it (?:grow|rise|fall|go up|go down|give)\b", r"\bmultibagger\b",
    r"\bpredict\b", r"\bforecast\b", r"\btarget price\b", r"\bwhen to (?:buy|sell|exit)\b",
]

# 3) Performance / returns computation we must not answer. -----------------------
PERFORMANCE_PATTERNS = [
    r"\breturns?\b", r"\bcagr\b", r"\byield\b", r"\bperformance\b",
    r"\bhow much (?:will|would|did) i (?:earn|get|make)\b",
    r"\bmaturity value\b", r"\bfuture value\b", r"\bcompare returns\b",
    r"\bwhich gave (?:more|higher|better) return\b", r"\bpast performance\b",
]

# Domain vocabulary — query must contain at least one of these to be in-scope. ----
DOMAIN_TERMS = {
    "expense", "ratio", "ter", "exit", "load", "sip", "minimum", "min",
    "lock", "lockin", "elss", "riskometer", "risk", "benchmark",
    "nifty", "statement", "gains", "download", "scheme", "fund", "funds",
    "hdfc", "flexi", "flexicap", "cap", "midcap", "mid", "largecap", "large",
    "tax", "saver", "nav", "plan", "lumpsum", "folio", "inception", "objective",
    "category", "cas", "consolidated", "invest", "investment",
}


def detect_pii(text: str) -> str | None:
    """Return the first PII type found in the text, else None."""
    for label, pattern in PII_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return None


def is_advice(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in ADVICE_PATTERNS)


def is_performance(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in PERFORMANCE_PATTERNS)


def in_domain(text: str) -> bool:
    terms = set(re.findall(r"[a-z]+", text.lower()))
    return bool(terms & DOMAIN_TERMS)
