# 📘 Facts-Only MF Assistant

A small **RAG-based FAQ chatbot** that answers **factual** questions about mutual fund
schemes using **only official public pages** from the AMC (HDFC Mutual Fund), **SEBI**
and **AMFI**. Every answer shows **one citation link**. It **refuses** opinion / advice /
portfolio questions, never makes performance claims, and never accepts or stores PII.

> **Facts-only. No investment advice.**

Built for **Milestone 4 — Mutual Fund FAQs (Facts-Only Q&A)**.
Stack: **Python + Streamlit** (UI) · **TF-IDF retrieval** (RAG) · **Google Gemini** (answer phrasing, free tier).

---

## Scope (corpus)

**AMC:** HDFC Mutual Fund · **Schemes (4):**

| Scheme | Category | Benchmark | Riskometer |
|---|---|---|---|
| HDFC Large Cap Fund (formerly HDFC Top 100 Fund) | Large Cap | NIFTY 100 TRI | Very High |
| HDFC Flexi Cap Fund | Flexi Cap | NIFTY 500 TRI | Very High |
| HDFC ELSS Tax Saver | ELSS (tax-saver) | NIFTY 500 TRI | Very High |
| HDFC Mid Cap Fund (formerly HDFC Mid-Cap Opportunities Fund) | Mid Cap | NIFTY Midcap 150 TRI | Very High |

Plus SEBI/AMFI concept pages for **expense ratio (TER), riskometer, ELSS lock-in**, and
HDFC **statement / capital-gains** how-to guides. The full corpus lives in
[`data/corpus.json`](data/corpus.json); the **22 source URLs** are in
[`data/sources.csv`](data/sources.csv) and listed below.

---

## What it answers (and what it won't)

**Answers (fact + 1 citation):** expense ratio · exit load · minimum SIP · ELSS lock-in ·
riskometer · benchmark · how to download account / capital-gains statements.

**Refuses, politely:**
- **Advice / opinion / portfolio** — "Should I buy…?", "Which is best?" → facts-only
  message + SEBI investor-education link.
- **Performance / returns** — "What returns will I get?" → no computation; points to the
  official factsheet.
- **PII** — a PAN, Aadhaar, phone, email, OTP or account/folio number in the query is
  detected, refused and **not stored**.
- **Out of scope** — anything not in the corpus → "I don't have that fact in my sources."

---

## How it works (architecture)

```
User query
   │
   ├─ Guardrail 1: PII detection (regex)              → refuse, don't store
   ├─ Guardrail 2: advice/opinion detection            → facts-only refusal + edu link
   ├─ Guardrail 3: performance/returns detection        → "see factsheet" refusal
   ├─ Domain gate: is the query about our corpus?       → else "not in sources"
   │
   ├─ Retrieval (RAG): TF-IDF cosine over corpus.json
   │        + lexical boost on topic keywords (picks expense-ratio vs benchmark vs lock-in)
   │
   └─ Generation: Gemini answers strictly from the top-k retrieved chunks
            (≤3 sentences, facts only)  ── fallback ──▶ deterministic extractive answer
   │
   └─ Output: answer + one "Source" link + "Last updated from sources: …"
```

- **W1 (Thinking like a model):** classify each query — answerable fact vs. refuse — before retrieving.
- **W2 (LLMs & prompting):** strict instruction prompt, ≤3-sentence concise answers, polite safe-refusals, consistent citation wording.
- **W3 (RAG):** small-corpus retrieval with accurate citations from AMC/SEBI/AMFI pages.

> **No Gemini key? It still works.** Without `GEMINI_API_KEY` the app runs in
> **extractive mode** — it returns the fact verbatim from the top-ranked cited source. The
> key only makes phrasing smoother; facts and citations are identical.

---

## Setup

### Run locally
```bash
pip install -r requirements.txt

# (optional, for Gemini phrasing) get a free key at https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="your-key"      # Windows: setx GEMINI_API_KEY "your-key"

streamlit run app.py
```
Open http://localhost:8501.

### Deploy free on Streamlit Community Cloud
1. Push this folder to a **public GitHub repo**.
2. Go to https://share.streamlit.io → **New app** → pick the repo, branch, and `app.py`.
3. **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
   (see [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example))
4. **Deploy** → share the resulting `*.streamlit.app` URL as the working-prototype link.

---

## Repo layout
```
facts-mf-assistant/
├─ app.py                      # Streamlit app: guardrails + RAG + Gemini
├─ data/
│  ├─ corpus.json              # curated fact chunks (with source_url + last_updated)
│  └─ sources.csv              # 22 official AMC/SEBI/AMFI URLs
├─ requirements.txt
├─ .streamlit/secrets.toml.example
├─ sample_qa.md                # 12 sample queries with answers + links
├─ disclaimer.md               # UI disclaimer snippet
└─ README.md
```

---

## Source list (22 official URLs)

**AMC — HDFC Mutual Fund**
1. HDFC Large Cap Fund — Regular: https://www.hdfcfund.com/product-solutions/overview/hdfc-top-100-fund/regular
2. HDFC Large Cap Fund — Direct: https://www.hdfcfund.com/product-solutions/overview/hdfc-top-100-fund/direct
3. HDFC Flexi Cap Fund — Regular: https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/regular
4. HDFC Flexi Cap Fund — Direct: https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct
5. HDFC ELSS Tax Saver — Regular: https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver/regular
6. HDFC ELSS Tax Saver — Direct: https://www.hdfcfund.com/explore/mutual-funds/hdfc-elss-tax-saver/direct
7. HDFC Mid Cap Fund — Regular: https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/regular
8. HDFC Mid Cap Fund — Direct: https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct
9. SID — HDFC Flexi Cap Fund: https://files.hdfcfund.com/s3fs-public/SID/2025-05/SID%20-%20HDFC%20Flexi%20Cap%20Fund%20dated%20May%2030,%202025.pdf
10. SID — HDFC ELSS Tax Saver Fund: https://files.hdfcfund.com/s3fs-public/SID/2024-11/SID%20-%20HDFC%20ELSS%20Tax%20Saver%20Fund%20dated%20November%2021,%202024.pdf
11. KIM — HDFC Mid-Cap Opportunities Fund: https://files.hdfcfund.com/s3fs-public/KIM/2024-11/KIM%20-%20HDFC%20Mid-Cap%20Opportunities%20Fund%20dated%20November%2021,%202024.pdf
12. Scheme Summary — HDFC Top 100 Fund: https://files.hdfcfund.com/s3fs-public/2023-06/HDFC%20Top%20100%20Fund.pdf
13. Total Expense Ratio — Notices: https://www.hdfcfund.com/statutory-disclosure/total-expense-ratio-of-mutual-fund-schemes/notices
14. Total Expense Ratio — Reports: https://www.hdfcfund.com/statutory-disclosure/total-expense-ratio-of-mutual-fund-schemes/reports
15. Download Consolidated Account Statement (CAS): https://www.hdfcfund.com/services/consolidated-account-statement
16. How to get a Capital Gain Statement: https://www.hdfcfund.com/learn/blog/how-get-capital-gain-statement-mutual-fund-schemes-india
17. Investor Services — KYC, Statements & Forms: https://www.hdfcfund.com/investor-services

**AMFI**
18. Knowledge Center — Expense Ratio (TER): https://www.amfiindia.com/investor/knowledge-center-info?zoneName=expenseRatio
19. Investor Corner: https://www.amfiindia.com/investor

**SEBI**
20. SEBI Investor — Understanding the Riskometer: https://investor.sebi.gov.in/riskometer.html
21. SEBI Investor — A Guide to ELSS: https://investor.sebi.gov.in/elss.html
22. TER & Performance Disclosure for Mutual Funds (Circular): https://www.sebi.gov.in/legal/circulars/oct-2018/total-expense-ratio-ter-and-performance-disclosure-for-mutual-funds_40766.html

---

## Sample Q&A

See [`sample_qa.md`](sample_qa.md) for 12 worked examples (8 factual + 4 refusals). Two examples:

> **Q: What is the expense ratio of HDFC Flexi Cap Fund?**
> A: The Total Expense Ratio (Regular Plan) is 1.37% (inclusive of GST on management fees).
> **Source:** https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/regular
> _Last updated from sources: 2026-09-29_

> **Q: Should I buy HDFC Flexi Cap Fund?**  *(refused)*
> A: I'm a facts-only assistant, so I can't give buy/sell or portfolio advice. … For
> guidance, please consult a SEBI-registered adviser.

---

## Disclaimer (as shown in the UI)

> **Facts-only. No investment advice.** This assistant shares publicly available facts
> about mutual fund schemes from official AMC/SEBI/AMFI pages. It does not recommend
> buying/selling/holding any scheme, does not compute or compare returns, and never
> accepts or stores PII (PAN, Aadhaar, account/folio numbers, OTPs, email, phone).
> Figures such as expense ratios change over time — verify against the linked official
> source. Mutual fund investments are subject to market risks; read all scheme related
> documents carefully. For personalised guidance, consult a SEBI-registered adviser.

(Full snippet: [`disclaimer.md`](disclaimer.md).)

---

## Known limits

- **Small, curated corpus** — 4 HDFC schemes + core concepts. Anything outside returns a
  polite "not in my sources" message.
- **Point-in-time facts** — figures (esp. TER) were compiled from official pages on
  **2026-09-29** and can change; the app links the source so users can verify the latest
  value, and points to HDFC's live TER page for current plan-wise ratios.
- **Regular-plan TER shown** by default; Direct-plan TER differs — see the linked TER page.
- **Retrieval is TF-IDF (lexical)**, not embeddings — great for this small corpus and fully
  offline-capable, but very unusual phrasings may need a scheme/fact keyword to match.
- **No live NAV / performance** by design (no performance claims).
- **No transactions / no login / no PII** — informational only.
