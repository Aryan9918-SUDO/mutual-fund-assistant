"""
Facts-Only MF Assistant  —  RAG-based FAQ chatbot for HDFC mutual fund schemes.

Answers ONLY factual questions (expense ratio, exit load, minimum SIP, ELSS lock-in,
riskometer, benchmark, how to download statements) from official AMC / SEBI / AMFI
sources. Every answer carries one citation link. It refuses opinion/advice questions
and never accepts or stores PII.

Skills demonstrated:
  W1 - Thinking like a model : classify each query (answerable fact vs. refuse).
  W2 - LLMs & prompting      : strict instruction prompt, concise phrasing, safe refusals.
  W3 - RAG                    : small-corpus TF-IDF retrieval with accurate citations.
"""

import json
import os
import re
from pathlib import Path

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------------- #
APP_TITLE = "Facts-Only MF Assistant"
CORPUS_PATH = Path(__file__).parent / "data" / "corpus.json"
GEMINI_MODEL = "gemini-2.0-flash"       # free-tier friendly; change if unavailable
TOP_K = 3                                # chunks retrieved per query
MIN_SIM = 0.05                           # below this we treat as "not in corpus"
SOURCES_LAST_UPDATED = "2026-09-29"     # date corpus was compiled from official pages

DISCLAIMER = (
    "Facts-only. No investment advice. This assistant shares publicly available "
    "facts about mutual fund schemes from official AMC/SEBI/AMFI pages and does not "
    "recommend buying, selling or holding any scheme."
)

EDU_LINK = "https://investor.sebi.gov.in/"   # SEBI investor education, used in refusals

EXAMPLE_QUESTIONS = [
    "What is the expense ratio of HDFC Flexi Cap Fund?",
    "What is the lock-in period for HDFC ELSS Tax Saver?",
    "How do I download my capital gains statement?",
]

# ----------------------------------------------------------------------------- #
# Guardrail patterns  (W1: decide answer vs. refuse)
# ----------------------------------------------------------------------------- #

# 1) PII we must never accept or store.
PII_PATTERNS = {
    "PAN": r"\b[A-Za-z]{5}[0-9]{4}[A-Za-z]\b",
    "Aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "phone number": r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b",
    "email address": r"\b[\w.\-]+@[\w\-]+\.[A-Za-z]{2,}\b",
    "OTP": r"\b(?:otp|one[\-\s]?time\s?password)\b[\s:is]*\d{4,8}\b",
    "account number": r"\b(?:a/c|acc(?:ount)?|folio)\s*(?:no\.?|number|#)?\s*[:\-]?\s*\d{6,}\b",
}

# 2) Opinion / advice / portfolio questions we must refuse.
ADVICE_PATTERNS = [
    r"\bshould i\b", r"\bshould we\b", r"\bis it (?:good|bad|worth|safe|better)\b",
    r"\bwhich (?:is )?(?:the )?best\b", r"\bbest (?:fund|scheme|option)\b",
    r"\brecommend\b", r"\bsuggest\b", r"\badvice\b", r"\badvise\b",
    r"\bbuy or sell\b", r"\bbuy\b", r"\bsell\b", r"\binvest in\b",
    r"\bworth (?:buying|investing|it)\b", r"\bgood (?:to|for) invest\b",
    r"\bwill it (?:grow|rise|fall|go up|go down|give)\b", r"\bmultibagger\b",
    r"\bpredict\b", r"\bforecast\b", r"\btarget price\b", r"\bwhen to (?:buy|sell|exit)\b",
]

# 3) Performance / returns computation we must not answer (no performance claims).
PERFORMANCE_PATTERNS = [
    r"\breturns?\b", r"\bcagr\b", r"\byield\b", r"\bperformance\b",
    r"\bhow much (?:will|would|did) i (?:earn|get|make)\b",
    r"\bmaturity value\b", r"\bfuture value\b", r"\bcompare returns\b",
    r"\bwhich gave (?:more|higher|better) return\b", r"\bpast performance\b",
]


def detect_pii(text: str):
    """Return the first PII type found, or None."""
    for label, pattern in PII_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return label
    return None


def is_advice(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in ADVICE_PATTERNS)


def is_performance(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in PERFORMANCE_PATTERNS)


# ----------------------------------------------------------------------------- #
# Retrieval  (W3: small-corpus RAG)
# ----------------------------------------------------------------------------- #
TOPIC_BOOST = 0.12   # bonus per query term that matches a chunk's topic keywords

# Query must contain at least one of these to be considered in-domain; otherwise we
# return a polite "not in my sources" message instead of a spurious lexical match.
DOMAIN_TERMS = {
    "expense", "ratio", "ter", "exit", "load", "sip", "minimum", "min",
    "lock", "lockin", "lock-in", "elss", "riskometer", "risk", "benchmark",
    "nifty", "statement", "gains", "download", "scheme", "fund", "funds",
    "hdfc", "flexi", "flexicap", "cap", "midcap", "mid", "largecap", "large",
    "tax", "saver", "nav", "plan", "lumpsum", "folio", "inception", "objective",
    "category", "cas", "consolidated", "invest", "investment",
}


def in_domain(query: str) -> bool:
    terms = set(re.findall(r"[a-z]+", query.lower()))
    return bool(terms & DOMAIN_TERMS)


@st.cache_resource(show_spinner=False)
def load_index():
    docs = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    # Index over scheme + topic + text so both the scheme name and the fact type
    # contribute to the TF-IDF match.
    corpus_text = [f"{d['scheme']} {d['topic']} {d['text']}" for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus_text)
    return docs, vectorizer, matrix


def retrieve(query: str, k: int = TOP_K):
    """Hybrid retrieval: TF-IDF cosine + a lexical boost when query terms match the
    chunk's topic keywords. Chunks of one scheme differ mainly by fact type, so the
    topic boost reliably picks 'expense ratio' vs 'benchmark' vs 'lock-in'."""
    docs, vectorizer, matrix = load_index()
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    q_terms = {w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2}
    scored = []
    for i, d in enumerate(docs):
        topic_terms = set(re.findall(r"[a-z]+", d["topic"].lower()))
        boost = TOPIC_BOOST * len(q_terms & topic_terms)
        scored.append((float(sims[i]) + boost, float(sims[i]), i))
    scored.sort(reverse=True)
    # Return only chunks with real TF-IDF signal (boost alone is not enough).
    return [(docs[i], base) for _, base, i in scored[:k] if base > 0]


# ----------------------------------------------------------------------------- #
# Generation  (W2: strict prompting; Gemini with extractive fallback)
# ----------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are "Facts-Only MF Assistant", a RAG chatbot answering factual \
questions about HDFC mutual fund schemes.

STRICT RULES:
- Answer ONLY using the CONTEXT below. If the context does not contain the answer, say \
you don't have that fact in your sources.
- Keep the answer to a maximum of 3 short sentences. Be concise and neutral.
- State facts only. NEVER give opinions, recommendations, or investment advice.
- NEVER predict, compute, or compare returns/performance.
- Do not add any citation text yourself; the app appends the citation separately.
- If asked in another tone, stay factual and polite.

CONTEXT:
{context}

USER QUESTION: {question}

Answer (<=3 sentences, facts only):"""


def get_api_key():
    # Streamlit Cloud: st.secrets; local: environment variable.
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


def generate_with_gemini(question: str, context: str):
    """Return answer text, or None if Gemini is unavailable (fall back to extractive)."""
    api_key = get_api_key()
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=(
                "You answer strictly from provided context, in <=3 sentences, "
                "facts only, no advice, no return predictions."
            ),
        )
        prompt = SYSTEM_PROMPT.format(context=context, question=question)
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 220},
        )
        return (resp.text or "").strip() or None
    except Exception as e:  # network/key/model errors -> graceful fallback
        st.session_state["_gemini_error"] = str(e)
        return None


def extractive_answer(query, hits):
    """Deterministic fallback (no Gemini): pick, from the top chunk, the sentences
    most relevant to the query by keyword overlap; cap at 3 sentences."""
    text = hits[0][0]["text"]
    # Split on sentence enders but not after the "Rs." abbreviation.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])(?<!Rs\.)\s+", text) if s.strip()]
    if len(sentences) <= 2:
        return " ".join(sentences)
    q_terms = {w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2}
    scored = []
    for idx, s in enumerate(sentences):
        s_terms = set(re.findall(r"[a-z]+", s.lower()))
        overlap = len(q_terms & s_terms)
        scored.append((overlap, -idx, s))  # ties -> earlier sentence first
    scored.sort(reverse=True)
    if scored[0][0] == 0:                   # nothing matched -> return lead sentences
        return " ".join(sentences[:2])
    chosen = [s for _, _, s in scored[:2]]
    # keep original order for readability
    chosen.sort(key=lambda s: sentences.index(s))
    return " ".join(chosen)


# ----------------------------------------------------------------------------- #
# Core answer pipeline
# ----------------------------------------------------------------------------- #
def answer_query(query: str) -> dict:
    """Return {'type': ..., 'text': ..., 'source': (name,url)|None}."""
    q = query.strip()
    if not q:
        return {"type": "empty", "text": "Please type a question.", "source": None}

    # --- Guardrail 1: PII (highest priority; never store/echo) ---
    pii = detect_pii(q)
    if pii:
        return {
            "type": "pii",
            "text": (
                f"For your safety I can't accept or process personal information such as "
                f"a {pii}. Please remove it and ask only about scheme facts "
                f"(e.g. expense ratio, exit load, lock-in). No PII is stored."
            ),
            "source": ("SEBI Investor education", EDU_LINK),
        }

    # --- Guardrail 2: advice / opinion / portfolio ---
    if is_advice(q):
        return {
            "type": "advice",
            "text": (
                "I'm a facts-only assistant, so I can't give buy/sell or portfolio advice. "
                "I can share factual details like expense ratio, exit load, benchmark, "
                "riskometer or lock-in. For guidance, please consult a SEBI-registered adviser."
            ),
            "source": ("SEBI Investor education", EDU_LINK),
        }

    # --- Guardrail 3: performance / returns computation ---
    if is_performance(q):
        return {
            "type": "performance",
            "text": (
                "I don't compute or compare returns or performance. Please refer to the "
                "scheme's official factsheet for past performance figures. I can share other "
                "facts like expense ratio, benchmark or minimum SIP."
            ),
            "source": ("HDFC Mutual Fund - Fund Factsheets", "https://www.hdfcfund.com/"),
        }

    # --- Retrieval + answerability check ---
    hits = retrieve(q)
    if not in_domain(q) or not hits or hits[0][1] < MIN_SIM:
        return {
            "type": "no_context",
            "text": (
                "I don't have that fact in my official sources. I currently cover HDFC "
                "Large Cap, Flexi Cap, ELSS Tax Saver and Mid Cap funds, plus SEBI/AMFI "
                "concepts like expense ratio, riskometer and ELSS lock-in."
            ),
            "source": None,
        }

    context = "\n\n".join(
        f"[{d['scheme']} | {d['topic']}] {d['text']}" for d, _ in hits
    )
    top_doc = hits[0][0]

    answer = generate_with_gemini(q, context) or extractive_answer(q, hits)
    return {
        "type": "answer",
        "text": answer,
        "source": (top_doc["source_name"], top_doc["source_url"]),
        "last_updated": top_doc.get("last_updated", SOURCES_LAST_UPDATED),
    }


# ----------------------------------------------------------------------------- #
# UI
# ----------------------------------------------------------------------------- #
def render_answer(res: dict):
    st.markdown(res["text"])
    if res.get("source"):
        name, url = res["source"]
        st.markdown(f"**Source:** [{name}]({url})")
    if res.get("type") == "answer":
        st.caption(f"Last updated from sources: {res.get('last_updated', SOURCES_LAST_UPDATED)}")


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📘", layout="centered")
    st.title("📘 " + APP_TITLE)
    st.write(
        "Ask factual questions about **HDFC mutual fund schemes** "
        "(Large Cap, Flexi Cap, ELSS Tax Saver, Mid Cap) — expense ratio, exit load, "
        "minimum SIP, ELSS lock-in, riskometer, benchmark, or how to download statements."
    )
    st.info("**Facts-only. No investment advice.**", icon="ℹ️")

    with st.expander("Try an example question"):
        cols = st.columns(len(EXAMPLE_QUESTIONS))
        for i, ex in enumerate(EXAMPLE_QUESTIONS):
            if cols[i].button(ex, key=f"ex_{i}"):
                st.session_state["pending"] = ex

    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Replay prior turns.
    for turn in st.session_state["history"]:
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            render_answer(turn["res"])

    prompt = st.chat_input("Ask a factual question about an HDFC scheme...")
    if "pending" in st.session_state and not prompt:
        prompt = st.session_state.pop("pending")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        res = answer_query(prompt)
        with st.chat_message("assistant"):
            render_answer(res)
        st.session_state["history"].append({"q": prompt, "res": res})

    with st.sidebar:
        st.subheader("About")
        st.write(DISCLAIMER)
        st.markdown("---")
        st.caption(
            "Sources: official HDFC Mutual Fund, SEBI and AMFI public pages only. "
            "See `data/sources.csv`."
        )
        if not get_api_key():
            st.warning(
                "No `GEMINI_API_KEY` found — running in offline **extractive** mode "
                "(answers are pulled verbatim from the cited source). Add the key in "
                "Streamlit secrets or as an environment variable to enable Gemini phrasing."
            )


if __name__ == "__main__":
    main()
