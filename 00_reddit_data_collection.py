#!/usr/bin/env python3
"""
00_data_collection.py
=====================
Collect health product claims from Reddit using public JSON endpoints.
No API key required.

Output: data/00_raw/reddit_posts.jsonl  (one JSON object per post)
        data/00_raw/all_posts.jsonl     (copy for downstream scripts)
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = Path("data/00_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUBREDDITS = [
    "supplements",
    "AlternativeHealth",
    "Nootropics",
    "herbalism",
    "NaturalRemedies",
    "nutrition",
    "Biohackers",
    "SkincareAddiction",
    "PCOS",
    "Fibromyalgia",
    "CancerFightersClub",
    "weightloss",
    "fasting",
    "keto",
]

HEALTH_CLAIM_KEYWORDS = [
    "cures", "heals", "treats", "prevents", "miracle",
    "detox", "cleanse", "anti-aging", "immune boost",
    "weight loss", "fat burner", "metabolism",
    "clinically proven", "scientifically proven",
    "big pharma", "natural remedy", "no side effects",
    "cancer", "diabetes", "depression", "anxiety",
    "inflammation", "blood pressure", "cholesterol",
    "turmeric", "ashwagandha", "ivermectin", "colloidal silver",
    "essential oil", "CBD", "kratom", "black seed oil",
    "apple cider vinegar", "hydrogen peroxide",
    "supplement", "vitamin", "zinc", "magnesium",
    "probiotic", "omega", "fish oil", "melatonin",
    "gut health", "hormone", "thyroid", "cortisol",
    "boost immunity", "immune system",
]

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) health_misinfo_research/1.0"


# ============================================================
# Reddit Public JSON (no credentials needed)
# ============================================================
def fetch_reddit_json(url: str) -> dict:
    """Fetch JSON from Reddit's public endpoint."""
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 429:
            logger.warning("Rate limited, waiting 60s...")
            time.sleep(60)
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        logger.warning(f"HTTP {e.code} for {url}")
        return {}
    except Exception as e:
        logger.warning(f"Request error: {e}")
        return {}


def collect_subreddit(sub_name: str, sort: str = "hot", limit: int = 100) -> list:
    """Collect posts from a subreddit using public JSON endpoint."""
    url = f"https://www.reddit.com/r/{sub_name}/{sort}.json?limit={limit}&raw_json=1"
    if sort == "top":
        url += "&t=year"

    data = fetch_reddit_json(url)
    if not data or "data" not in data:
        return []

    posts = []
    for child in data.get("data", {}).get("children", []):
        pd = child.get("data", {})
        text = f"{pd.get('title', '')} {pd.get('selftext', '')}".lower()

        if not any(kw.lower() in text for kw in HEALTH_CLAIM_KEYWORDS):
            continue

        post = {
            "id": pd.get("id", ""),
            "platform": "reddit",
            "subreddit": sub_name,
            "title": pd.get("title", ""),
            "text": pd.get("selftext", "")[:5000],
            "score": pd.get("score", 0),
            "num_comments": pd.get("num_comments", 0),
            "created_utc": pd.get("created_utc", 0),
            "url": f"https://reddit.com{pd.get('permalink', '')}",
            "author": pd.get("author", "[deleted]"),
            "collected_at": datetime.utcnow().isoformat(),
        }
        posts.append(post)

    return posts


def collect_comments(sub_name: str, post_id: str) -> list:
    """Collect top-level comments from a post."""
    url = f"https://www.reddit.com/r/{sub_name}/comments/{post_id}.json?limit=20&raw_json=1"
    data = fetch_reddit_json(url)
    if not data or len(data) < 2:
        return []

    comments = []
    for child in data[1].get("data", {}).get("children", []):
        if child.get("kind") != "t1":
            continue
        cd = child.get("data", {})
        ctext = cd.get("body", "").lower()
        if any(kw.lower() in ctext for kw in HEALTH_CLAIM_KEYWORDS):
            comments.append({
                "id": cd.get("id", ""),
                "platform": "reddit",
                "subreddit": sub_name,
                "title": "",
                "text": cd.get("body", "")[:3000],
                "score": cd.get("score", 0),
                "num_comments": 0,
                "created_utc": cd.get("created_utc", 0),
                "url": f"https://reddit.com{cd.get('permalink', '')}",
                "author": cd.get("author", "[deleted]"),
                "collected_at": datetime.utcnow().isoformat(),
            })
    return comments


def collect_reddit(output_path: Path):
    """Collect posts from all health-related subreddits."""
    all_posts = []

    for sub_name in SUBREDDITS:
        sub_posts = []
        for sort_method in ["hot", "new", "top"]:
            logger.info(f"Collecting r/{sub_name}/{sort_method}...")
            try:
                posts = collect_subreddit(sub_name, sort=sort_method, limit=100)
                sub_posts.extend(posts)
                time.sleep(3)
            except Exception as e:
                logger.warning(f"  Error r/{sub_name}/{sort_method}: {e}")
                time.sleep(5)

        # Collect comments from top 5 posts
        top_posts = [p for p in sub_posts if p.get("score", 0) > 5][:5]
        for tp in top_posts:
            try:
                coms = collect_comments(sub_name, tp["id"])
                sub_posts.extend(coms)
                time.sleep(2)
            except Exception as e:
                logger.debug(f"  Comment error: {e}")

        all_posts.extend(sub_posts)
        logger.info(f"  r/{sub_name}: {len(sub_posts)} posts collected")

    # Deduplicate
    seen = set()
    unique_posts = []
    for p in all_posts:
        if p["id"] and p["id"] not in seen:
            seen.add(p["id"])
            unique_posts.append(p)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for post in unique_posts:
            f.write(json.dumps(post, ensure_ascii=False) + "\n")
    logger.info(f"Collected {len(unique_posts)} unique Reddit posts -> {output_path}")

    # Copy as all_posts.jsonl
    all_path = OUTPUT_DIR / "all_posts.jsonl"
    with open(all_path, "w", encoding="utf-8") as f:
        for post in unique_posts:
            f.write(json.dumps(post, ensure_ascii=False) + "\n")
    logger.info(f"Copied to {all_path}")

    return unique_posts


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Stage 00: Data Collection (Reddit public JSON)")
    logger.info("=" * 60)

    reddit_output = OUTPUT_DIR / "reddit_posts.jsonl"
    collect_reddit(reddit_output)
    logger.info("Done.")
