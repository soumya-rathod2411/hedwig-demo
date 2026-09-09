"""
hedwig_collector.py

Now uses Tavily (an official search API built for AI agents) instead
of the free/unofficial DuckDuckGo scraper -- more accurate, structured
results, and no more surprise rate-limit blocks.

Budget: 10 categories x 1 query each, run 3x/day (every 8 hours) =
30 queries/day, ~900/month -- comfortably inside Tavily's free
1,000/month tier.

Requires:
    pip install requests
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # same fix as the reporter -- non-ASCII in a title/error shouldn't crash printing

# =========================================================
# Reads from environment variables first (how GitHub Actions passes in
# secrets), falling back to the hardcoded value below for local testing.
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "tvly-your_key_here").strip()
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "your_youtube_api_key_here").strip()
# =========================================================

DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)  # creates the folder if it doesn't exist yet

# 3 query VARIANTS per category, rotated day by day (see query_for_today
# below). Instead of asking the exact same narrow question forever, each
# day explores a different angle of the same category -- this is what
# actually broadens coverage over time instead of just repeating.
CATEGORY_QUERY_VARIANTS = {
    "AI": [
        ("latest AI model release agentic AI news", "news"),
        ("AI research breakthrough new paper", "news"),
        ("AI coding assistant new feature update", "general"),
    ],
    "Developer World": [
        ("new programming framework GitHub trending open source", "general"),
        ("new database tool developer release", "general"),
        ("backend API deployment DevOps new tool", "general"),
    ],
    "Cybersecurity": [
        ("critical vulnerability CVE data breach news", "news"),
        ("new cybersecurity attack technique trend", "news"),
        ("CERT-In advisory security patch India", "news"),
    ],
    "India Policy": [
        ("IndiaAI Mission MeitY Digital India policy news", "news"),
        ("India data protection cybersecurity regulation update", "news"),
        ("India government technology scheme announcement", "news"),
    ],
    "Learning": [
        ("free online course certification computer science", "general"),
        ("new free certification AI cloud course", "general"),
        ("free university course programming YouTube", "general"),
    ],
    "Opportunities": [
        ("hackathon internship India students technology", "general"),
        ("scholarship fellowship tech students India", "general"),
        ("open source program student developer India", "general"),
    ],
    "Developer Tools": [
        ("new developer tool AI coding assistant cloud free tier", "general"),
        ("new IDE extension developer productivity tool", "general"),
        ("free API SDK developer platform launch", "general"),
    ],
    "Hardware": [
        ("semiconductor GPU AI chip news", "news"),
        ("new AI accelerator hardware release", "news"),
        ("edge computing ARM chip announcement", "news"),
    ],
    "Major Companies": [
        ("OpenAI Anthropic Google Microsoft NVIDIA announcement", "news"),
        ("big tech AI product launch partnership", "news"),
        ("tech company acquisition funding AI", "news"),
    ],
    "World Tech Context": [
        ("technology geopolitics chips supply chain regulation", "news"),
        ("global technology regulation internet policy", "news"),
        ("semiconductor industry global development", "news"),
    ],
}

# Which categories also get a YouTube search. Learning included now, so
# course videos specifically get pulled in, not just written articles.
YOUTUBE_CATEGORIES = ["AI", "Developer Tools", "Learning"]

# Rotates which variant is used each day -- day 0 uses variant 0, day 1
# uses variant 1, day 2 uses variant 2, day 3 wraps back to variant 0.
VARIANT_INDEX_TODAY = datetime.now().toordinal() % 3

def query_for_today(category):
    query, topic = CATEGORY_QUERY_VARIANTS[category][VARIANT_INDEX_TODAY]
    return query, topic

def search_tavily(query, topic="general", max_results=5):
    """
    Calls Tavily's search API directly. Returns a list of results,
    each with title, content (a clean summary, not raw HTML), url,
    and published date when available.
    """
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "topic": topic,
            "max_results": max_results
        }
    )
    response.raise_for_status()  # raises an error if the request failed
    return response.json().get("results", [])

def search_youtube(query, max_results=3):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet", "q": query, "type": "video",
        "order": "date", "maxResults": max_results, "key": YOUTUBE_API_KEY
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []
    items = response.json().get("items", [])
    return [{
        "title": i["snippet"]["title"],
        "channel": i["snippet"]["channelTitle"],
        "description": i["snippet"]["description"]
    } for i in items]

def with_retry(func, max_attempts=3, wait_seconds=10):
    """Same idea as before: retry on failure with an increasing wait."""
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            if attempt < max_attempts:
                current_wait = wait_seconds * attempt
                print(f"  [failed (attempt {attempt}/{max_attempts}), retrying in {current_wait}s: {e}]")
                time.sleep(current_wait)
            else:
                print(f"  [failed after {max_attempts} attempts, giving up: {e}]")
    return None


# --- Dedup: remember what's already been surfaced recently ---
SEEN_FILE = os.path.join(DATA_FOLDER, "hedwig_seen.json")
SEEN_RETENTION_DAYS = 14  # forget items older than this -- a story can resurface after enough time

def normalize_title(title):
    """Lowercase + collapse whitespace, so near-identical titles from
    different sources still match as 'the same story' for dedup purposes."""
    return re.sub(r"\s+", " ", title.strip().lower())

def load_seen():
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_seen(seen):
    tmp_path = SEEN_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(seen, f, indent=2)
    os.replace(tmp_path, SEEN_FILE)

def prune_seen(seen):
    cutoff = datetime.now() - timedelta(days=SEEN_RETENTION_DAYS)
    return {title: ts for title, ts in seen.items() if datetime.fromisoformat(ts) > cutoff}

seen = prune_seen(load_seen())

# --- Collect across all 10 categories ---
timestamp = datetime.now().isoformat()
collected = []
skipped_duplicates = 0

for category in CATEGORY_QUERY_VARIANTS:
    query, topic = query_for_today(category)
    print(f"Searching [{category}] via Tavily ({topic}), variant {VARIANT_INDEX_TODAY}: \"{query}\"")

    results = with_retry(lambda: search_tavily(query, topic))
    if results is None:
        collected.append({"timestamp": timestamp, "type": "search_failed", "category": category, "source": "tavily"})
    else:
        for r in results:
            title = r.get("title", "")
            norm = normalize_title(title)
            if norm and norm in seen:
                skipped_duplicates += 1
                continue  # already reported on within the last SEEN_RETENTION_DAYS -- skip it
            collected.append({
                "timestamp": timestamp,
                "type": "tavily",
                "category": category,
                "title": title,
                "body": r.get("content", ""),
                "source": r.get("url", ""),
                "published": r.get("published_date", "")
            })
            if norm:
                seen[norm] = datetime.now().isoformat()
    time.sleep(1)  # Tavily is a proper API -- much lighter delay needed than the old scraper

    if category in YOUTUBE_CATEGORIES:
        try:
            for item in search_youtube(query):
                title = item.get("title", "")
                norm = normalize_title(title)
                if norm and norm in seen:
                    skipped_duplicates += 1
                    continue
                collected.append({"timestamp": timestamp, "type": "youtube", "category": category, **item})
                if norm:
                    seen[norm] = datetime.now().isoformat()
        except Exception as e:
            print(f"  [YouTube search failed for {category}, skipping: {e}]")
        time.sleep(1)

save_seen(seen)

# --- Append to ONE ongoing data file (not tied to calendar date) ---
filename = os.path.join(DATA_FOLDER, "hedwig_data.jsonl")

with open(filename, "a", encoding="utf-8") as f:
    for entry in collected:
        f.write(json.dumps(entry) + "\n")

print(f"\n[{timestamp}] Collected {len(collected)} new items across {len(CATEGORY_QUERY_VARIANTS)} categories "
      f"({skipped_duplicates} duplicates skipped) into {filename}")
