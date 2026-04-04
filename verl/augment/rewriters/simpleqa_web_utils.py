"""Shared helpers for SimpleQA web-based rewriters."""

from __future__ import annotations

import json
import re
from html import unescape
from typing import Dict, List, Optional, Sequence
from urllib.parse import unquote, urlparse

import requests

_USER_AGENT = "verl-simpleqa-web-search/0.1"


def extract_urls(sample: Dict) -> List[str]:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        metadata = sample.get("metadat")
    if not isinstance(metadata, dict):
        return []
    urls = metadata.get("urls")
    if not isinstance(urls, list):
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def is_wikipedia_url(url: str) -> bool:
    parsed = urlparse(url)
    if "wikipedia.org" not in parsed.netloc:
        return False
    return "/wiki/" in parsed.path


def _wikipedia_title(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if not is_wikipedia_url(url):
        return None
    path = parsed.path
    if "/wiki/" not in path:
        return None
    title = path.split("/wiki/", 1)[1]
    title = unquote(title)
    return title or None


def fetch_wikipedia_extract(url: str, timeout: int) -> Optional[str]:
    title = _wikipedia_title(url)
    if not title:
        return None
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "explaintext": 1,
        "exsectionformat": "plain",
        "format": "json",
    }
    headers = {"User-Agent": _USER_AGENT}
    resp = requests.get(api, params=params, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None
    page = next(iter(pages.values()))
    extract = page.get("extract")
    if not isinstance(extract, str):
        return None
    return extract.strip() or None


def _strip_html(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw_html)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|li|h\d)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[\t\r\f\v ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def fetch_html_text(url: str, timeout: int, max_chars: int) -> Optional[str]:
    headers = {"User-Agent": _USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return None
    text = _strip_html(resp.text)
    if not text:
        return None
    return text[:max_chars]


def pick_source_text(urls: Sequence[str], timeout: int, max_chars: int) -> tuple[Optional[str], Optional[str]]:
    wiki_urls = [u for u in urls if is_wikipedia_url(u)]
    for url in wiki_urls:
        try:
            extract = fetch_wikipedia_extract(url, timeout)
        except Exception:
            extract = None
        if extract:
            return extract, url

    for url in urls:
        try:
            text = fetch_html_text(url, timeout, max_chars)
        except Exception:
            text = None
        if text:
            return text, url
    return None, None


_MIN_CONTENT_CHARS = 150
# Wikipedia pages can be large; fetch more so local search can scan the full article.
_WIKI_MAX_CHARS = 300_000


def iter_source_texts(urls: Sequence[str], timeout: int, max_chars: int):
    """Yield (content, url) for each URL that returns usable content, wiki URLs first.

    Wikipedia pages are fetched via HTML (preserving tables) with a large char limit
    so that local answer search can scan the full article. Non-wiki pages use max_chars.
    Skips content shorter than _MIN_CONTENT_CHARS to filter junk. Deduplicates URLs.
    """
    seen: set[str] = set()
    ordered = [u for u in urls if is_wikipedia_url(u)] + [u for u in urls if not is_wikipedia_url(u)]
    for url in ordered:
        if url in seen:
            continue
        seen.add(url)
        try:
            limit = _WIKI_MAX_CHARS if is_wikipedia_url(url) else max_chars
            content = fetch_html_text(url, timeout, limit)
        except Exception:
            content = None
        if content and len(content) >= _MIN_CONTENT_CHARS:
            yield content, url


def _chars_around(text: str, pos: int, match_len: int, window_chars: int) -> str:
    half = window_chars // 2
    start = max(0, pos - half)
    end = min(len(text), pos + match_len + half)
    return text[start:end].strip()


def find_answer_window(text: str, answer: str, window_chars: int = 3000) -> Optional[str]:
    """Return a passage from text containing the answer, using local string search.

    Tries exact match first, then token-based scoring (useful for unit variants
    like '390m' vs '390 metres', or partial names like 'Hapke' for 'Bruce W. Hapke').
    Returns None if no reasonable match is found.
    """
    if not text or not answer:
        return None

    text_lower = text.lower()
    answer_lower = answer.lower().strip()

    # 1. Exact substring match
    pos = text_lower.find(answer_lower)
    if pos != -1:
        return _chars_around(text, pos, len(answer_lower), window_chars)

    # 2. Token-based: score positions by how many answer tokens appear nearby
    tokens = [t for t in re.findall(r"\b\w+\b", answer_lower) if len(t) > 3]
    if not tokens:
        return None

    # Collect all positions where any token appears
    candidate_positions: List[int] = []
    for token in tokens:
        start = 0
        while True:
            p = text_lower.find(token, start)
            if p == -1:
                break
            candidate_positions.append(p)
            start = p + 1

    if not candidate_positions:
        return None

    # Pick the position whose surrounding window contains the most tokens
    best_pos = None
    best_score = 0
    for center in candidate_positions:
        region_start = max(0, center - window_chars // 2)
        region_end = min(len(text_lower), center + window_chars // 2)
        region = text_lower[region_start:region_end]
        score = sum(1 for t in tokens if t in region)
        if score > best_score:
            best_score = score
            best_pos = center

    # Require at least half the tokens to be present
    if best_pos is None or best_score < max(1, len(tokens) // 2):
        return None

    return _chars_around(text, best_pos, 0, window_chars)


def parse_json_payload(payload: str) -> Optional[dict]:
    if not isinstance(payload, str):
        return None
    # Try raw parse first
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    stripped = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", payload, flags=re.DOTALL).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Extract first {...} block
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
