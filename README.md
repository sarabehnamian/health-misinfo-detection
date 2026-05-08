# Health Misinformation Detection Pipeline

## LLM-grounded detection of health product misinformation on social media

This repository contains the **seven Python modules** below and a short note under `data/README.md`. **Raw and processed social posts (Reddit, YouTube, etc.) are not distributed here** — they may be subject to platform terms, copyright, or privacy rules; run the collectors locally if your use is permitted.

### Repository scripts (complete list)

| File | Role |
|------|------|
| `00_reddit_data_collection.py` | Reddit: public JSON (`r/{sub}/hot|new|top.json`), keyword filter, optional top-level comments from high-score posts → `reddit_posts.jsonl` and `all_posts.jsonl`. |
| `00b_youtube_collection.py` | YouTube Data API v3: search queries → videos → keyword-filtered comments (and qualifying video metadata) → `youtube_posts.jsonl`, merge/dedupe into `all_posts.jsonl`. |
| `01_claim_extraction.py` | LLM extracts structured claims from each post line in `all_posts.jsonl` (or `reddit_posts.jsonl` fallback) → `extracted_claims.jsonl`. |
| `01b_resume_extraction.py` | Same extraction prompt/model as batch 01; reads **`data/00_raw/remaining_posts.jsonl`** only and **appends** to `extracted_claims.jsonl`; stops on Anthropic credit errors. |
| `02_evidence_retrieval.py` | For each claim: PubMed Entrez search/fetch, openFDA **drug adverse events** (`drug/event.json`), NIH ODS/NCCIH URL map → `claims_with_evidence.jsonl`. |
| `03_veracity_classification.py` | LLM+RAG classification + **keyword baseline**; writes `classification.llm_grounded` and `classification.keyword_baseline` per line → `classified_claims.jsonl`. |
| `04b_publication_figures.py` | Reads classified claims; writes `descriptive_stats.json`, figures **fig1–fig7**, and `examples_<LABEL>.txt`. (Module docstring inside the file still says `04_publication_figures.py`; the real filename is **`04b_publication_figures.py`**.) |

---

### Overview

The pipeline: (1) collect Reddit and optionally YouTube text, (2) extract many structured “claims” per post via Claude, (3) attach PubMed/FDA/NIH pointers as evidence, (4) classify each claim with the same evidence in the prompt (RAG) plus a simple keyword baseline, (5) summarize and plot.

---

### Stage 00 — Reddit (`00_reddit_data_collection.py`)

- **No API keys:** `urllib` to `https://www.reddit.com/.../*.json?raw_json=1` with a fixed `User-Agent`.
- **Subreddits:** 14 communities (e.g. `supplements`, `Nootropics`, `nutrition`, `keto`, … — see `SUBREDDITS` in the file).
- **Per subreddit:** collects `hot`, `new`, and `top` (top uses `t=year`), up to **100** items per listing; **3 s** sleep between listings.
- **Keyword gate:** post/comment text must match at least one string in `HEALTH_CLAIM_KEYWORDS`.
- **Comments:** for up to **5** posts with score **> 5**, fetches top-level comments (`limit=20` on the thread JSON); **2 s** sleep between comment fetches.
- **Outputs:** `data/00_raw/reddit_posts.jsonl` and the same deduplicated stream copied to `data/00_raw/all_posts.jsonl`.

### Stage 00b — YouTube (`00b_youtube_collection.py`)

- **Requires:** `YOUTUBE_API_KEY`, `pip install google-api-python-client`.
- **Search:** fixed list of `SEARCH_QUERIES`; up to **`MAX_VIDEOS_PER_QUERY` (10)** videos per query; **0.5 s** between queries.
- **Comments:** up to **`MAX_COMMENTS_PER_VIDEO` (50)** per video, relevance order; **0.3 s** between videos.
- **Records:** keyword-matched video title+description as synthetic posts (`id` like `yt_vid_...`) and keyword-matched comments (`yt_com_...`); `platform` is `"youtube"`.
- **Outputs:** `data/00_raw/youtube_posts.jsonl`; **merges** new IDs into existing `all_posts.jsonl` (keeps prior lines if file exists).

---

### Stage 01 — Claim extraction (`01_claim_extraction.py`)

- **Input (in order):** `data/00_raw/all_posts.jsonl`, else `data/00_raw/reddit_posts.jsonl`.
- **Skip:** combined title+text shorter than **20** characters.
- **LLM default:** Anthropic `claude-haiku-4-5-20251001`, `max_tokens=2000`, **0.5 s** between posts.
- **Optional OpenAI:** `extract_claims_with_openai` uses `gpt-4o`; switch by changing `llm_provider` in `if __name__ == "__main__"` (default is `"anthropic"`).

**Schema produced by the model (one JSON object per claim, one line per claim in `extracted_claims.jsonl`):**

| Field | Meaning |
|-------|---------|
| `product` | Substance or product named |
| `claimed_effect` | Alleged effect |
| `target_condition` | Condition or `general health` |
| `claim_strength` | `definitive`, `suggestive`, or `anecdotal` |
| `verbatim_quote` | ≤100 words from the post |
| `risk_category` | `supplement_efficacy`, `drug_misuse`, `disease_cure`, `weight_loss`, `detox`, `anti_aging`, `mental_health`, `other` |

**Provenance added in code:** `source_post_id`, `source_platform`, `source_subreddit`, `source_url`, `post_score`.

### Stage 01b — Resume (`01b_resume_extraction.py`)

- **Input:** only `data/00_raw/remaining_posts.jsonl` (you must create/split this file yourself).
- **Output:** **append** mode to `data/01_claims/extracted_claims.jsonl`.
- **Anthropic only** (no OpenAI branch); same model as 01; stops if the API error string contains `400` and `credit`.

---

### Stage 02 — Evidence (`02_evidence_retrieval.py`)

- **PubMed:** query = product + condition (if not `general health`) else product + effect; search string requires systematic review **or** meta-analysis **or** RCT publication types; fetches details via `efetch`; **0.4 s** sleep after PubMed per claim.
- **NCBI:** `email` and `tool` query parameters are set in code (`ENTREZ_EMAIL` in the file — replace for your own runs if needed).
- **openFDA:** `https://api.fda.gov/drug/event.json` search on `patient.drug.medicinalproduct`, **limit 3**; **0.3 s** sleep after FDA per claim. *(The Python function is named `search_fda_warnings`; it queries adverse events, not FDA warning letters.)*
- **NIH:** static `NIH_ODS_SUPPLEMENTS` map → optional `nih_reference` object with `reference_url` and `note`.
- **Stored under each claim as** `evidence`: `pubmed_articles`, `fda_events`, `nih_reference`, `pubmed_query`, `evidence_count`.

**Note:** The module docstring at the top of `02_evidence_retrieval.py` still mentions DrugBank and warning letters; **those are not implemented** in the current code.

---

### Stage 03 — Veracity (`03_veracity_classification.py`)

- **Input:** `data/02_evidence/claims_with_evidence.jsonl`  
- **Output:** `data/03_classified/classified_claims.jsonl`  
- **LLM+RAG:** Anthropic `claude-haiku-4-5-20251001` by default (`max_tokens=1000`), **0.5 s** between claims; optional OpenAI `gpt-4o` via `llm_provider` in `main`.
- **Each output line** adds:

```json
"classification": {
  "llm_grounded": {
    "veracity": "SUPPORTED|UNSUPPORTED|EXAGGERATED|CONTRADICTED|DANGEROUS",
    "confidence": 0.0,
    "risk_tier": "LOW|MODERATE|HIGH|CRITICAL",
    "reasoning": "...",
    "key_evidence": "...",
    "recommendation": "..."
  },
  "keyword_baseline": {
    "veracity": "DANGEROUS|EXAGGERATED|UNSUPPORTED",
    "confidence": 0.0,
    "risk_tier": "...",
    "reasoning": "...",
    "method": "keyword_baseline"
  }
}
```

The keyword baseline **never** assigns `SUPPORTED` or `CONTRADICTED` (only dangerous phrases, exaggeration phrases, or default `UNSUPPORTED`). It scores `claimed_effect` + `verbatim_quote` against `DANGEROUS_KEYWORDS` and `EXAGGERATION_KEYWORDS` in the source file.

---

### Stage 04b — Figures and stats (`04b_publication_figures.py`)

- **Input:** `data/03_classified/classified_claims.jsonl`
- **Output directory:** `data/04_evaluation/results/`
- **Writes:** `descriptive_stats.json`; `fig1_veracity_distribution.png`; `fig2_llm_vs_baseline.png`; `fig3_risk_distribution.png`; `fig4_platform_comparison.png` (expects `reddit` and `youtube` keys in the data); `fig5_category_heatmap.png`; `fig6_confidence_boxplot.png`; `fig7_category_distribution.png`; `examples_<VERACITY>.txt` for each of the five LLM labels (up to 20 examples each, sorted by confidence).
- **Plot style:** matplotlib **Agg**, Set3-inspired colors, **200 DPI**, serif fonts (see `setup()` in code).

---

### Requirements

```bash
pip install anthropic openai numpy matplotlib
# Optional: YouTube
pip install google-api-python-client
```

- **Python:** 3.9+ recommended  
- **Secrets:** `ANTHROPIC_API_KEY` (default pipeline); `OPENAI_API_KEY` if you switch 01/03 to OpenAI; `YOUTUBE_API_KEY` for 00b  
- **Network:** NCBI Entrez, openFDA, LLM APIs  

---

### Usage (run in order)

```bash
python 00_reddit_data_collection.py
python 00b_youtube_collection.py          # optional
python 01_claim_extraction.py
python 01b_resume_extraction.py         # optional; needs remaining_posts.jsonl
python 02_evidence_retrieval.py
python 03_veracity_classification.py
python 04b_publication_figures.py
```

---

### Data layout (created locally; not committed)

The `data/` tree is listed in `.gitignore`. After you run the stages, you should see something like:

```
data/
├── 00_raw/
│   ├── reddit_posts.jsonl
│   ├── all_posts.jsonl
│   ├── youtube_posts.jsonl              # after 00b
│   └── remaining_posts.jsonl            # user-built; for 01b only
├── 01_claims/
│   └── extracted_claims.jsonl
├── 02_evidence/
│   └── claims_with_evidence.jsonl
├── 03_classified/
│   └── classified_claims.jsonl
└── 04_evaluation/
    └── results/
        ├── descriptive_stats.json
        ├── fig1_veracity_distribution.png … fig7_category_distribution.png
        ├── fig2_llm_vs_baseline.png
        └── examples_<VERACITY>.txt
```

PNG files appear after you run `04b_publication_figures.py`. Aggregated metrics are written to `descriptive_stats.json` in the same results folder (no third-party post text in that file).

---

### Authors

1. Sara Behnamian — Globe Institute, University of Copenhagen  
2. Zeinab Shahbazi — Kristianstad University, Sweden  
3. Zahra Shahbazi — University of Padova, Italy  
4. Sadiqa Jafari — Gachon University, South Korea  

### License

This project is released under the [MIT License](LICENSE).
