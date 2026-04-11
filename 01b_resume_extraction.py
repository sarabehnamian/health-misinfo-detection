#!/usr/bin/env python3
"""
01b_resume_extraction.py
========================
Process remaining posts (2300+) and APPEND to existing extracted_claims.jsonl
Stops immediately if credits run out.
"""

import json
import time
import os
import logging
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROMPT = """You are a health misinformation analyst. Extract ALL health claims from the following social media post.

For each claim, provide a JSON object with these fields:
- "product": the supplement, drug, or substance mentioned
- "claimed_effect": what the product allegedly does
- "target_condition": the health condition targeted, or "general health"
- "claim_strength": one of "definitive", "suggestive", or "anecdotal"
- "verbatim_quote": the exact phrase from the post containing the claim (max 100 words)
- "risk_category": one of "supplement_efficacy", "drug_misuse", "disease_cure", "weight_loss", "detox", "anti_aging", "mental_health", "other"

If the post contains NO health claims, return: {"claims": []}

Return ONLY valid JSON with the structure: {"claims": [list of claim objects]}

POST:
"""

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
posts = [json.loads(l) for l in open("data/00_raw/remaining_posts.jsonl", encoding="utf-8")]
logger.info(f"Processing {len(posts)} remaining posts...")
total = 0

with open("data/01_claims/extracted_claims.jsonl", "a", encoding="utf-8") as fout:
    for i, post in enumerate(posts):
        text = f"{post.get('title', '')} {post.get('text', '')}".strip()
        if not text or len(text) < 20:
            continue
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                messages=[{"role": "user", "content": PROMPT + text[:3000]}],
            )
            t = resp.content[0].text.strip()
            if t.startswith("```"):
                t = t.split("```")[1]
                if t.startswith("json"):
                    t = t[4:]
                t = t.strip()
            result = json.loads(t)
            for claim in result.get("claims", []):
                claim["source_post_id"] = post["id"]
                claim["source_platform"] = post["platform"]
                claim["source_subreddit"] = post.get("subreddit", "")
                claim["source_url"] = post.get("url", "")
                claim["post_score"] = post.get("score", 0)
                fout.write(json.dumps(claim, ensure_ascii=False) + "\n")
                fout.flush()
                total += 1
        except Exception as e:
            if "400" in str(e) and "credit" in str(e).lower():
                logger.error(f"OUT OF CREDITS at post {i}. {total} new claims saved. STOPPING.")
                break
            logger.warning(f"Error: {e}")
        if (i + 1) % 50 == 0:
            logger.info(f"  {i+1}/{len(posts)}, {total} new claims")
        time.sleep(0.5)

logger.info(f"Done. {total} new claims appended to extracted_claims.jsonl")
