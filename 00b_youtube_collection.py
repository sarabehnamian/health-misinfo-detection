#!/usr/bin/env python3
"""
00b_youtube_collection.py
=========================
Collect health product claims from YouTube video comments.
Uses YouTube Data API v3 (free, 10,000 quota units/day).

Requires: pip install google-api-python-client
Set: YOUTUBE_API_KEY environment variable

Output: data/00_raw/youtube_posts.jsonl
        Merges into data/00_raw/all_posts.jsonl
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/00_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Health-related search queries for YouTube
SEARCH_QUERIES = [
    "turmeric cures cancer",
    "natural remedy diabetes",
    "apple cider vinegar health benefits",
    "supplement stack review",
    "colloidal silver health",
    "ivermectin covid treatment",
    "black seed oil cure",
    "detox cleanse results",
    "ashwagandha benefits",
    "vitamin D immune system",
    "CBD oil cures",
    "kratom health benefits",
    "hydrogen peroxide therapy",
    "herbal remedy cholesterol",
    "essential oils healing",
    "gut health supplement",
    "weight loss supplement review",
    "big pharma natural cure",
    "probiotics cure depression",
    "zinc cold flu prevention",
]

# Keywords to filter comments
HEALTH_CLAIM_KEYWORDS = [
    "cures", "heals", "treats", "prevents", "miracle",
    "detox", "cleanse", "immune boost", "weight loss",
    "clinically proven", "scientifically proven",
    "big pharma", "natural remedy", "no side effects",
    "cancer", "diabetes", "depression", "anxiety",
    "inflammation", "blood pressure", "cholesterol",
    "turmeric", "ashwagandha", "ivermectin", "colloidal silver",
    "essential oil", "CBD", "kratom", "black seed oil",
    "apple cider vinegar", "hydrogen peroxide",
    "supplement", "vitamin", "zinc", "magnesium",
    "probiotic", "omega", "fish oil", "melatonin",
    "gut health", "hormone", "boost immunity",
]

MAX_COMMENTS_PER_VIDEO = 50
MAX_VIDEOS_PER_QUERY = 10


def get_youtube_client():
    """Initialize YouTube API client."""
    import os
    try:
        from googleapiclient.discovery import build
    except ImportError:
        logger.error("Install: pip install google-api-python-client")
        return None

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        logger.error("Set YOUTUBE_API_KEY environment variable")
        return None

    return build("youtube", "v3", developerKey=api_key)


def search_videos(youtube, query: str, max_results: int = 10) -> list:
    """Search YouTube for health-related videos."""
    try:
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results,
            relevanceLanguage="en",
            order="relevance",
        )
        response = request.execute()

        videos = []
        for item in response.get("items", []):
            videos.append({
                "video_id": item["id"]["videoId"],
                "title": item["snippet"]["title"],
                "channel": item["snippet"]["channelTitle"],
                "published_at": item["snippet"]["publishedAt"],
                "description": item["snippet"].get("description", ""),
            })
        return videos

    except Exception as e:
        logger.warning(f"Search error for '{query}': {e}")
        return []


def get_video_comments(youtube, video_id: str, max_comments: int = 50) -> list:
    """Get comments from a YouTube video."""
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100),
            order="relevance",
            textFormat="plainText",
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
                "text": snippet.get("textDisplay", ""),
                "author": snippet.get("authorDisplayName", ""),
                "like_count": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt", ""),
            })

    except Exception as e:
        # Comments might be disabled
        logger.debug(f"Comment error for {video_id}: {e}")

    return comments


def collect_youtube(output_path: Path):
    """Collect health-related YouTube comments."""
    youtube = get_youtube_client()
    if not youtube:
        return []

    all_posts = []
    seen_ids = set()

    for query in SEARCH_QUERIES:
        logger.info(f"Searching YouTube: '{query}'...")
        videos = search_videos(youtube, query, max_results=MAX_VIDEOS_PER_QUERY)
        time.sleep(0.5)

        for video in videos:
            vid = video["video_id"]

            # Collect video title+description as a post
            video_text = f"{video['title']} {video['description']}"
            if any(kw.lower() in video_text.lower() for kw in HEALTH_CLAIM_KEYWORDS):
                post_id = f"yt_vid_{vid}"
                if post_id not in seen_ids:
                    seen_ids.add(post_id)
                    all_posts.append({
                        "id": post_id,
                        "platform": "youtube",
                        "subreddit": "",
                        "title": video["title"],
                        "text": video["description"][:5000],
                        "score": 0,
                        "num_comments": 0,
                        "created_utc": video.get("published_at", ""),
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "author": video["channel"],
                        "collected_at": datetime.utcnow().isoformat(),
                    })

            # Collect comments
            comments = get_video_comments(youtube, vid, MAX_COMMENTS_PER_VIDEO)
            time.sleep(0.3)

            for comment in comments:
                ctext = comment["text"].lower()
                if any(kw.lower() in ctext for kw in HEALTH_CLAIM_KEYWORDS):
                    post_id = f"yt_com_{comment['comment_id']}"
                    if post_id not in seen_ids:
                        seen_ids.add(post_id)
                        all_posts.append({
                            "id": post_id,
                            "platform": "youtube",
                            "subreddit": "",
                            "title": video["title"],
                            "text": comment["text"][:3000],
                            "score": comment.get("like_count", 0),
                            "num_comments": 0,
                            "created_utc": comment.get("published_at", ""),
                            "url": f"https://www.youtube.com/watch?v={vid}",
                            "author": comment.get("author", "[unknown]"),
                            "collected_at": datetime.utcnow().isoformat(),
                        })

        logger.info(f"  '{query}': {len(videos)} videos, {len(all_posts)} total posts so far")

    # Write YouTube posts
    with open(output_path, "w", encoding="utf-8") as f:
        for post in all_posts:
            f.write(json.dumps(post, ensure_ascii=False) + "\n")
    logger.info(f"Collected {len(all_posts)} YouTube posts -> {output_path}")

    # Merge with existing all_posts.jsonl
    all_path = OUTPUT_DIR / "all_posts.jsonl"
    existing_posts = []
    if all_path.exists():
        with open(all_path, encoding="utf-8") as f:
            for line in f:
                existing_posts.append(json.loads(line))
        logger.info(f"Existing posts in all_posts.jsonl: {len(existing_posts)}")

    # Deduplicate by ID
    existing_ids = {p["id"] for p in existing_posts}
    new_posts = [p for p in all_posts if p["id"] not in existing_ids]
    combined = existing_posts + new_posts

    with open(all_path, "w", encoding="utf-8") as f:
        for post in combined:
            f.write(json.dumps(post, ensure_ascii=False) + "\n")
    logger.info(f"Merged: {len(combined)} total posts in {all_path} ({len(new_posts)} new from YouTube)")

    return all_posts


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Stage 00b: YouTube Data Collection")
    logger.info("=" * 60)

    output = OUTPUT_DIR / "youtube_posts.jsonl"
    collect_youtube(output)
    logger.info("Done.")
