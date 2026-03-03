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


def parse_json_payload(payload: str) -> Optional[dict]:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None
