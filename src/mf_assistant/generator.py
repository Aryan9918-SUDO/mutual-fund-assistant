"""Answer generation (W2: LLMs & prompting).

Gemini generates a concise, grounded answer from the retrieved chunks. If no API key is
configured (or the call fails), we fall back to a deterministic *extractive* answer taken
verbatim from the top chunk — so the app always works, even offline.
"""
from __future__ import annotations

import re
from typing import List, Optional

from . import config
from .models import RetrievedChunk

SYSTEM_INSTRUCTION = (
    "You answer strictly from the provided context, in <=3 sentences, facts only, "
    "no advice, no return/performance predictions. If the context lacks the answer, "
    "say you don't have that fact in your sources."
)

PROMPT_TEMPLATE = """Answer the user's question using ONLY the CONTEXT below.

Rules:
- Maximum 3 short sentences. Be concise and neutral.
- State facts only. No opinions, recommendations or investment advice.
- Never predict, compute or compare returns/performance.
- Do not add citations yourself; the app appends the source separately.

CONTEXT:
{context}

QUESTION: {question}

Answer (<=3 sentences, facts only):"""


def build_context(chunks: List[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.scheme} | {c.topic}] {c.text}" for c in chunks)


def generate(question: str, chunks: List[RetrievedChunk]) -> str:
    """Return a grounded answer; Gemini if available, else extractive fallback."""
    return _generate_gemini(question, chunks) or extractive_answer(question, chunks)


def _generate_gemini(question: str, chunks: List[RetrievedChunk]) -> Optional[str]:
    api_key = config.get_gemini_api_key()
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            config.GEMINI_MODEL, system_instruction=SYSTEM_INSTRUCTION
        )
        prompt = PROMPT_TEMPLATE.format(context=build_context(chunks), question=question)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": config.GEN_TEMPERATURE,
                "max_output_tokens": config.GEN_MAX_TOKENS,
            },
        )
        return (resp.text or "").strip() or None
    except Exception:
        return None


def extractive_answer(question: str, chunks: List[RetrievedChunk]) -> str:
    """Pick the sentences from the top chunk most relevant to the query (<=3)."""
    text = chunks[0].text
    # Split on sentence enders but not after the "Rs." abbreviation.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])(?<!Rs\.)\s+", text) if s.strip()]
    if len(sentences) <= 2:
        return " ".join(sentences)
    q_terms = {w for w in re.findall(r"[a-z]+", question.lower()) if len(w) > 2}
    scored = [
        (len(q_terms & set(re.findall(r"[a-z]+", s.lower()))), -i, s)
        for i, s in enumerate(sentences)
    ]
    scored.sort(reverse=True)
    if scored[0][0] == 0:                       # nothing matched -> lead sentences
        return " ".join(sentences[:2])
    chosen = [s for _, _, s in scored[:2]]
    chosen.sort(key=sentences.index)            # keep natural order
    return " ".join(chosen)
