"""Rewrite SimpleQA samples using web-sourced passages."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from html import unescape
from typing import Dict, List, Optional, Sequence
from urllib.parse import unquote, urlparse

import requests

from ..openai_client import OpenAIClient
from ..registry import register
from ..schemas import attach_augmentation_metadata, get_prompt_text

_USER_AGENT = "verl-simpleqa-web-search/0.1"


def _extract_urls(sample: Dict) -> List[str]:
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        metadata = sample.get("metadat")
    if not isinstance(metadata, dict):
        return []
    urls = metadata.get("urls")
    if not isinstance(urls, list):
        return []
    return [u for u in urls if isinstance(u, str) and u.strip()]


def _is_wikipedia_url(url: str) -> bool:
    parsed = urlparse(url)
    if "wikipedia.org" not in parsed.netloc:
        return False
    return "/wiki/" in parsed.path


def _wikipedia_title(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if not _is_wikipedia_url(url):
        return None
    path = parsed.path
    if "/wiki/" not in path:
        return None
    title = path.split("/wiki/", 1)[1]
    title = unquote(title)
    return title or None


def _fetch_wikipedia_extract(url: str, timeout: int) -> Optional[str]:
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


def _fetch_html_text(url: str, timeout: int, max_chars: int) -> Optional[str]:
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


def _pick_source_text(urls: Sequence[str], timeout: int, max_chars: int) -> tuple[Optional[str], Optional[str]]:
    wiki_urls = [u for u in urls if _is_wikipedia_url(u)]
    for url in wiki_urls:
        try:
            extract = _fetch_wikipedia_extract(url, timeout)
        except Exception:
            extract = None
        if extract:
            return extract, url

    for url in urls:
        try:
            text = _fetch_html_text(url, timeout, max_chars)
        except Exception:
            text = None
        if text:
            return text, url
    return None, None


def _parse_json_payload(payload: str) -> Optional[dict]:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


@register("simpleqa_web_search")
class SimpleqaWebSearchRewriter:
    name = "simpleqa_web_search"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        max_chars: int = 12000,
        timeout: int = 30,
        log_path: str = "simpleqa_web_search.log",
        max_log_samples: int = 50,
    ) -> None:
        self._client = OpenAIClient(model)
        self._max_chars = max_chars
        self._timeout = timeout
        self._log_path = log_path
        self._max_log_samples = max_log_samples
        self._logged_samples = 0
        with open(self._log_path, "w", encoding="utf-8") as handle:
            handle.write("")

    def _log(self, message: str) -> None:
        if self._logged_samples >= self._max_log_samples:
            return
        with open(self._log_path, "a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def _truncate(self, text: str, max_len: int = 2000) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _extract_window(self, *, content: str, question: str, answer: str, seed: int | None) -> Optional[str]:
        system_prompt = (
            "You are a careful extractor. Return JSON only with keys 'status' and 'window'. "
            "If you can find the answer in the text, return status 'found' and the window of text around the answer."
            "If you cannot find the answer in the text, return status 'not_found' and window ''."
        )
        user_prompt = (
            "Find the span of text that contains the answer. "
            "Return up to 400 words before and 400 words after that span, VERBATIM, "
            "separated by a blank line from the answer span. If there are fewer words, return what exists.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Passage:\n"
            f"{content}"
        )
        raw = self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            seed=seed,
        )
        self._log("[extract_window] system_prompt=" + self._truncate(system_prompt))
        self._log("[extract_window] user_prompt=" + self._truncate(user_prompt))
        self._log("[extract_window] raw_response=" + self._truncate(raw))
        payload = _parse_json_payload(raw)
        if not payload:
            return None
        status = payload.get("status")
        if status not in ("ok", "found"):
            return None
        window = payload.get("window")
        if not isinstance(window, str):
            return None
        window = window.strip()
        return window or None

    def _window_sufficient(self, *, window: str, question: str, answer: str, seed: int | None) -> bool:
        system_prompt = (
            "You are a strict judge. Return JSON only with key 'sufficient' as true or false."
        )
        user_prompt = (
            "Given the passage, decide if the answer can be fully supported by it.\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Passage:\n"
            f"{window}"
        )
        raw = self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            seed=seed,
        )
        self._log("[window_sufficient] system_prompt=" + self._truncate(system_prompt))
        self._log("[window_sufficient] user_prompt=" + self._truncate(user_prompt))
        self._log("[window_sufficient] raw_response=" + self._truncate(raw))
        payload = _parse_json_payload(raw)
        if not payload or "sufficient" not in payload:
            return False
        return bool(payload.get("sufficient"))

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> List[dict]:
        self._logged_samples += 1
        self._log("=" * 40)
        self._log(f"[sample] seed={rng_seed}")

        urls = _extract_urls(sample)
        if not urls:
            self._log("[sample] no urls")
            return []
        self._log("[sample] urls=" + ", ".join(urls))

        content, source_url = _pick_source_text(urls, self._timeout, self._max_chars)
        if not content or not source_url:
            self._log("[sample] no content fetched")
            return []
        self._log(f"[sample] source_url={source_url}")
        self._log("[sample] content_preview=" + self._truncate(content))

        question = sample.get("question")
        if not isinstance(question, str) or not question.strip():
            question = get_prompt_text(sample)

        answer = sample.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            reward_model = sample.get("reward_model")
            if isinstance(reward_model, dict):
                answer = reward_model.get("ground_truth")
        if not isinstance(answer, str) or not answer.strip():
            self._log("[sample] missing answer")
            return []

        window = self._extract_window(
            content=content,
            question=question.strip(),
            answer=answer.strip(),
            seed=rng_seed,
        )
        if not window:
            self._log("[sample] no window extracted")
            return []
        self._log("[sample] window_preview=" + self._truncate(window))

        if not self._window_sufficient(
            window=window,
            question=question.strip(),
            answer=answer.strip(),
            seed=rng_seed,
        ):
            self._log("[sample] window insufficient")
            return []
        self._log("[sample] window sufficient")

        updated = deepcopy(sample)
        prompt_text = f"Write the following Wikipedia-style passage verbatim:\n\n{window}"
        if "prompt" in updated:
            updated["prompt"][0]["content"] = prompt_text
        if "question" in updated:
            updated["question"] = "Write the following Wikipedia-style passage verbatim:"
        if "answer" in updated:
            updated["answer"] = window
        reward_model = updated.get("reward_model")
        if isinstance(reward_model, dict) and "ground_truth" in reward_model:
            reward_model["ground_truth"] = window

        updated = attach_augmentation_metadata(
            updated,
            method_name=self.name,
            variant_idx=0,
            params={
                "model": self._client.model,
                "source_url": source_url,
                "max_chars": self._max_chars,
            },
            seed=rng_seed,
        )
        return [updated]
