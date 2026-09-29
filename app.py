"""Facts-Only MF Assistant — Streamlit UI.

Thin presentation layer over the `mf_assistant` package: all logic (guardrails, retrieval,
generation) lives in src/. This file only handles rendering and interaction.
"""
import json
import sys
from pathlib import Path

# Make the src-layout package importable when running on Streamlit Cloud (no pip install -e).
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from mf_assistant import Pipeline
from mf_assistant.config import (
    AMC_NAME, DISCLAIMER, EXAMPLE_QUESTIONS, SOURCES_LAST_UPDATED, get_gemini_api_key,
)
from mf_assistant.models import Answer, AnswerKind

SCHEME_FACTS = json.loads(
    (Path(__file__).resolve().parent / "data" / "scheme_facts.json").read_text("utf-8")
)


@st.cache_resource(show_spinner="Loading model & building index...")
def get_pipeline() -> Pipeline:
    return Pipeline()


def confidence_badge(conf: float | None) -> str:
    if conf is None:
        return ""
    if conf >= 0.6:
        return f"🟢 high confidence ({conf:.2f})"
    if conf >= 0.4:
        return f"🟡 medium confidence ({conf:.2f})"
    return f"🟠 low confidence ({conf:.2f})"


def render_answer(ans: Answer) -> None:
    st.markdown(ans.text)
    if ans.source_url:
        st.markdown(f"**Source:** [{ans.source_name}]({ans.source_url})")
    if ans.kind == AnswerKind.ANSWER:
        meta = f"Last updated from sources: {ans.last_updated or SOURCES_LAST_UPDATED}"
        badge = confidence_badge(ans.confidence)
        st.caption(meta + ("  ·  " + badge if badge else ""))
        if ans.retrieved:
            with st.expander("🔎 Sources I searched (retrieved evidence)"):
                for c in ans.retrieved:
                    st.markdown(
                        f"- **{c.scheme}** — _{c.topic}_  "
                        f"(similarity `{c.score:.3f}`)  \n  "
                        f"{c.text}  \n  [{c.source_name}]({c.source_url})"
                    )


def ask_tab() -> None:
    st.write(
        f"Ask factual questions about **{AMC_NAME}** schemes "
        "(Large Cap, Flexi Cap, ELSS Tax Saver, Mid Cap) — expense ratio, exit load, "
        "minimum SIP, ELSS lock-in, riskometer, benchmark, or how to download statements."
    )
    st.info("**Facts-only. No investment advice.**", icon="ℹ️")

    st.caption("Try an example:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, ex in enumerate(EXAMPLE_QUESTIONS):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["pending"] = ex

    st.session_state.setdefault("history", [])
    for turn in st.session_state["history"]:
        with st.chat_message("user"):
            st.markdown(turn["q"])
        with st.chat_message("assistant"):
            render_answer(turn["a"])

    prompt = st.chat_input("Ask a factual question about an HDFC scheme...")
    if "pending" in st.session_state and not prompt:
        prompt = st.session_state.pop("pending")
    # Shareable deep link: /?q=your+question auto-asks it once.
    if not prompt and not st.session_state["history"]:
        deep = st.query_params.get("q")
        if deep:
            prompt = deep

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        ans = get_pipeline().answer(prompt)
        with st.chat_message("assistant"):
            render_answer(ans)
        st.session_state["history"].append({"q": prompt, "a": ans})


def compare_tab() -> None:
    st.write("Compare **facts** across schemes side by side (no performance/returns).")
    names = list(SCHEME_FACTS["schemes"].keys())
    chosen = st.multiselect("Pick schemes to compare", names, default=names[:3])
    if not chosen:
        st.info("Select at least one scheme.")
        return
    cols = SCHEME_FACTS["columns"]
    table = {"Fact": cols}
    for name in chosen:
        s = SCHEME_FACTS["schemes"][name]
        table[name] = [s[c] for c in cols]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("Sources:")
    for name in chosen:
        s = SCHEME_FACTS["schemes"][name]
        st.markdown(f"- **{name}** → [{s['source_url']}]({s['source_url']})")
    st.caption(f"Last updated from sources: {SOURCES_LAST_UPDATED}")


def main() -> None:
    st.set_page_config(page_title="Facts-Only MF Assistant", page_icon="📘", layout="centered")
    st.title("📘 Facts-Only MF Assistant")

    tab_ask, tab_compare = st.tabs(["💬 Ask", "📊 Compare schemes"])
    with tab_ask:
        ask_tab()
    with tab_compare:
        compare_tab()

    with st.sidebar:
        st.subheader("About")
        st.write(DISCLAIMER)
        st.markdown("---")
        st.markdown(
            "**How it works**  \n"
            "1. Guardrails classify the query (advice / performance / PII).  \n"
            "2. Semantic retrieval (embeddings + FAISS) finds the fact.  \n"
            "3. Gemini writes a ≤3-sentence answer, grounded in one cited source."
        )
        st.caption("Sources: official HDFC / SEBI / AMFI public pages only (`data/sources.csv`).")
        if not get_gemini_api_key():
            st.warning(
                "No `GEMINI_API_KEY` set — running in offline **extractive** mode "
                "(answers taken verbatim from the cited source). Add the key in Streamlit "
                "secrets to enable Gemini phrasing."
            )


if __name__ == "__main__":
    main()
