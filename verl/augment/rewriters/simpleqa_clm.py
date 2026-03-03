"""Rewrite SimpleQA samples into text-only CLM rows using web passages."""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

from ..openai_client import OpenAIClient
from ..registry import register
from ..schemas import attach_augmentation_metadata, get_prompt_text
from .simpleqa_web_utils import extract_urls, parse_json_payload, pick_source_text


@register("simpleqa_clm")
class SimpleqaClmRewriter:
    name = "simpleqa_clm"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        max_chars: int = 12000,
        timeout: int = 30,
        log_path: str | None = None,
        max_log_samples: int = 0,
    ) -> None:
        self._client = OpenAIClient(model)
        self._max_chars = max_chars
        self._timeout = timeout
        self._log_path = log_path
        self._max_log_samples = max_log_samples
        self._logged_samples = 0
        self._metrics = {
            "total_samples": 0,
            "skipped_no_urls": 0,
            "skipped_no_content": 0,
            "skipped_missing_answer": 0,
            "skipped_no_window": 0,
            "skipped_window_insufficient": 0,
            "success": 0,
        }
        if self._log_path:
            with open(self._log_path, "w", encoding="utf-8") as handle:
                handle.write("")

    def _log(self, message: str) -> None:
        if not self._log_path or self._logged_samples >= self._max_log_samples:
            return
        with open(self._log_path, "a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def _truncate(self, text: str, max_len: int = 2000) -> str:
        if len(text) <= max_len:
            return text
        return text[:max_len] + "..."

    def _inc(self, key: str) -> None:
        if key in self._metrics:
            self._metrics[key] += 1

    def get_metrics(self) -> dict:
        return dict(self._metrics)

    def _extract_window(self, *, content: str, question: str, answer: str, seed: int | None) -> Optional[str]:
        system_prompt = (
            "You are a careful extractor. Return JSON only with keys 'status' and 'window'. "
            "ONLY return status 'found' only if the answer text appears as an exact substring in the passage. "
            "If you cannot find the answer in the text, return status 'not_found' and window ''."
        )
        user_prompt = (
            "Find the span of text that contains the answer, if it exists. "
            "Return up to 400 words before and 400 words after that span, VERBATIM, "
            "separated by a blank line from the answer span. If there are fewer words, return what exists.\n\n"
            "### EXAMPLE ###"
            "Example (do NOT mark as found):\n"
            "Passage excerpt: '... M.H. Beg ... was appointed Chief Justice of India by the Indira Gandhi government.'\n"
            "Question: Who appointed the Chief Justice of India, Mirza Hameedullah Beg, in 1977?\n"
            "Answer: Fakhruddin Ali Ahmed\n"
            "Because the answer name is not present, status must be 'not_found'.\n\n"
            "### END EXAMPLE ###\n"
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
        payload = parse_json_payload(raw)
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
            "You are a strict judge. Return JSON only with key 'sufficient' as true or false. "
            "ONLY set sufficient=true only if the answer is stated exactly as a substring in the passage. "
            "If the passage only implies the answer without stating it, sufficient must be false."
        )
        user_prompt = (
            "Given the passage, decide if the answer can be fully supported by it.\n\n"
            "### EXAMPLE ###\n\n"
            "Example (sufficient=false):\n"
            "Passage excerpt: '... appointed Chief Justice of India by the Indira Gandhi government.'\n"
            "Question: Who appointed the Chief Justice of India, Mirza Hameedullah Beg, in 1977?\n"
            "Answer: Fakhruddin Ali Ahmed\n\n"
            "### END EXAMPLE ###\n\n"
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
        payload = parse_json_payload(raw)
        if not payload or "sufficient" not in payload:
            return False
        return bool(payload.get("sufficient"))

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> List[dict]:
        self._inc("total_samples")
        self._logged_samples += 1
        self._log("=" * 40)
        self._log(f"[sample] seed={rng_seed}")

        urls = extract_urls(sample)
        if not urls:
            self._log("[sample] no urls")
            self._inc("skipped_no_urls")
            return []
        self._log("[sample] urls=" + ", ".join(urls))

        content, source_url = pick_source_text(urls, self._timeout, self._max_chars)
        if not content or not source_url:
            self._log("[sample] no content fetched")
            self._inc("skipped_no_content")
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
            self._inc("skipped_missing_answer")
            return []

        window = self._extract_window(
            content=content,
            question=question.strip(),
            answer=answer.strip(),
            seed=rng_seed,
        )
        if not window:
            self._log("[sample] no window extracted")
            self._inc("skipped_no_window")
            return []
        self._log("[sample] window_preview=" + self._truncate(window))

        if not self._window_sufficient(
            window=window,
            question=question.strip(),
            answer=answer.strip(),
            seed=rng_seed,
        ):
            self._log("[sample] window insufficient")
            self._inc("skipped_window_insufficient")
            return []
        self._log("[sample] window sufficient")

        updated = deepcopy(sample)
        updated["text"] = window

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
        self._inc("success")
        return [updated]
