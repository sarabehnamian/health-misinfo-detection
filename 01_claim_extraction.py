#!/usr/bin/env python3
"""
01_claim_extraction.py
======================
Extract structured health claims from raw social media posts using an LLM.

Input:  data/00_raw/all_posts.jsonl
Output: data/01_claims/extracted_claims.jsonl
"""

import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_DIR = Path("data/00_raw")
OUTPUT_DIR = Path("data/01_claims")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LLM Claim Extraction Prompt
# ============================================================
EXTRACTION_PROMPT_TEMPLATE = """You are a health misinformation analyst. Extract ALL health claims from the following social media post. 

For each claim, provide a JSON object with these fields:
- "product": the supplement, drug, or substance mentioned (e.g., "turmeric", "ivermectin", "apple cider vinegar")
- "claimed_effect": what the product allegedly does (e.g., "cures cancer", "lowers blood pressure", "boosts immunity")
- "target_condition": the health condition targeted (e.g., "cancer", "hypertension", "obesity"), or "general health" if nonspecific
- "claim_strength": one of "definitive" (stated as fact), "suggestive" (implies benefit), or "anecdotal" (personal story)
- "verbatim_quote": the exact phrase from the post containing the claim (max 100 words)
- "risk_category": one of "supplement_efficacy", "drug_misuse", "disease_cure", "weight_loss", "detox", "anti_aging", "mental_health", "other"

If the post contains NO health claims, return: {"claims": []}

Return ONLY valid JSON with the structure: {"claims": [list of claim objects]}

POST:
"""


# ============================================================
# LLM API Call (Anthropic Claude)
# ============================================================
def extract_claims_with_llm(post_text: str, model: str = "claude-haiku-4-5-20251001") -> dict:
    """
    Send post to Claude API for claim extraction.
    """
    import os

    try:
        from anthropic import Anthropic
    except ImportError:
        logger.error("Install anthropic: pip install anthropic")
        return {"claims": []}

    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = EXTRACTION_PROMPT_TEMPLATE + post_text[:3000]

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Handle potential markdown code fences
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        result = json.loads(text)
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")
        return {"claims": []}
    except Exception as e:
        logger.warning(f"API error: {e}")
        return {"claims": []}


# ============================================================
# Alternative: OpenAI GPT-4
# ============================================================
def extract_claims_with_openai(post_text: str, model: str = "gpt-4o") -> dict:
    """
    Alternative: use OpenAI GPT-4 for extraction.
    """
    import os

    try:
        from openai import OpenAI
    except ImportError:
        logger.error("Install openai: pip install openai")
        return {"claims": []}

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = EXTRACTION_PROMPT_TEMPLATE + post_text[:3000]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You extract health claims from social media posts. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=2000,
        )

        text = response.choices[0].message.content.strip()
        return json.loads(text)

    except Exception as e:
        logger.warning(f"OpenAI error: {e}")
        return {"claims": []}


# ============================================================
# Batch Processing
# ============================================================
def process_all_posts(input_path: Path, output_path: Path, llm_provider: str = "anthropic"):
    """Process all posts and extract claims."""

    extract_fn = extract_claims_with_llm if llm_provider == "anthropic" else extract_claims_with_openai

    posts = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            posts.append(json.loads(line))

    logger.info(f"Processing {len(posts)} posts with {llm_provider}...")

    total_claims = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, post in enumerate(posts):
            text = f"{post.get('title', '')} {post.get('text', '')}".strip()
            if not text or len(text) < 20:
                continue

            result = extract_fn(text)
            claims = result.get("claims", [])

            for claim in claims:
                claim["source_post_id"] = post["id"]
                claim["source_platform"] = post["platform"]
                claim["source_subreddit"] = post.get("subreddit", "")
                claim["source_url"] = post.get("url", "")
                claim["post_score"] = post.get("score", 0)
                fout.write(json.dumps(claim, ensure_ascii=False) + "\n")
                total_claims += 1

            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{len(posts)} posts, {total_claims} claims extracted")

            time.sleep(0.5)  # Rate limiting

    logger.info(f"Extracted {total_claims} claims from {len(posts)} posts -> {output_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Stage 01: Claim Extraction")
    logger.info("=" * 60)

    input_path = INPUT_DIR / "all_posts.jsonl"
    output_path = OUTPUT_DIR / "extracted_claims.jsonl"

    if not input_path.exists():
        input_path = INPUT_DIR / "reddit_posts.jsonl"

    if not input_path.exists():
        logger.error("No input file found. Run 00_reddit_data_collection.py (and optionally 00b_youtube_collection.py) first.")
        exit(1)

    process_all_posts(input_path, output_path, llm_provider="anthropic")
