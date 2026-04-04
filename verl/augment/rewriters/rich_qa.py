"""Rewrite SimpleQA samples by generating richer Q/A from web passages."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from ..openai_client import OpenAIClient
from ..registry import register
from ..schemas import attach_augmentation_metadata, get_prompt_text
from .simpleqa_web_utils import extract_urls, find_answer_window, iter_source_texts, parse_json_payload


@register("rich_qa")
class RichQaRewriter:
    name = "rich_qa"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        max_chars: int = 6000,
        timeout: int = 60,
        log_path: str | None = None,
        max_log_samples: int | None = None,
    ) -> None:
        self._client = OpenAIClient(model, timeout=float(timeout))
        self._max_chars = max_chars
        self._timeout = timeout
        self._max_log_samples = max_log_samples
        self._logged_samples = 0
        normalized_log_path = Path(log_path) if log_path else Path("rich_qa.log")
        normalized_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = normalized_log_path
        self._failure_dir = normalized_log_path.parent / f"{normalized_log_path.stem}_failures"
        self._failure_dir.mkdir(parents=True, exist_ok=True)
        for f in self._failure_dir.glob("*.log"):
            f.unlink()
        self._metrics = {
            "total_samples": 0,
            "skipped_no_urls": 0,
            "skipped_no_content": 0,
            "skipped_missing_answer": 0,
            "skipped_no_window": 0,
            "skipped_rich_failed": 0,
            "skipped_verification_failed": 0,
            "success": 0,
        }
        with open(self._log_path, "w", encoding="utf-8") as handle:
            handle.write("")

    def _log(self, message: str) -> None:
        if self._max_log_samples is not None and self._logged_samples > self._max_log_samples:
            return
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def _extract_sample_id(self, sample: dict) -> Optional[str]:
        extra_info = sample.get("extra_info")
        if isinstance(extra_info, dict):
            sample_id = extra_info.get("sample_id")
            if isinstance(sample_id, str):
                return sample_id
        sample_id = sample.get("sample_id")
        if isinstance(sample_id, str):
            return sample_id
        return None

    def _log_failure(
        self,
        failure_type: str,
        sample: dict,
        *,
        question: str | None = None,
        answer: str | None = None,
        details: str | None = None,
    ) -> None:
        if not self._failure_dir:
            return
        parts: list[str] = [failure_type]
        sample_id = self._extract_sample_id(sample)
        if sample_id:
            parts.append(f"sample_id={sample_id}")
        if question:
            parts.append(f"question={self._truncate(question, 200)}")
        if answer:
            parts.append(f"answer={self._truncate(answer, 200)}")
        if details:
            parts.append(details)
        entry = " | ".join(parts)
        failure_path = self._failure_dir / f"{failure_type}.log"
        with failure_path.open("a", encoding="utf-8") as handle:
            handle.write(entry + "\n")

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
            "Return status 'found' if the passage contains information that clearly supports the answer — "
            "including abbreviations, alternate spellings, unit variations (e.g. '390m' for '390 metres'), "
            "partial names (e.g. 'Hapke' for 'Bruce W. Hapke'), or paraphrased equivalents. "
            "Only return status 'not_found' if the passage genuinely does not contain the answer or any clear reference to it."
        )
        user_prompt = (
            "Find the span of text that most clearly supports the answer, if it exists. "
            "Return up to 400 words before and 400 words after that span, VERBATIM, "
            "separated by a blank line from the answer span. If there are fewer words, return what exists.\n\n"
            "### EXAMPLE (do NOT mark as found) ###\n"
            "Passage excerpt: '... M.H. Beg ... was appointed Chief Justice of India by the Indira Gandhi government.'\n"
            "Question: Who appointed the Chief Justice of India, Mirza Hameedullah Beg, in 1977?\n"
            "Answer: Fakhruddin Ali Ahmed\n"
            "Because neither 'Fakhruddin Ali Ahmed' nor any clear reference to him is present, status must be 'not_found'.\n\n"
            "### EXAMPLE (DO mark as found) ###\n"
            "Passage excerpt: '... the bridge spans 390m across the river ...'\n"
            "Question: What is the length of the Abdullah Bridge?\n"
            "Answer: 390 metres\n"
            "Because '390m' clearly corresponds to '390 metres', status must be 'found'.\n\n"
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

    def _generate_rich_qa(
        self,
        *,
        passage: str,
        question: str,
        answer: str,
        seed: int | None,
    ) -> Optional[dict]:
        system_prompt = (
            "You generate richer Q/A pairs grounded in the passage. "
            "Return JSON only with keys 'rich_question' and 'rich_answer'."
        )
        user_prompt = (
            "Given a passage and an original question/answer, produce a richer questiown and answer that are fully "
            "answerable from the passage. The rich question should ask for more detail or a broader slice of facts "
            "that includes the original answer. The rich answer should be concise and directly supported by the passage.\n\n"
            "### EXAMPLE 1 ###\n"
            "[Wikipedia Passage]\n"
            "The IEEE Frank Rosenblatt Award is a Technical Field Award established by the Institute of Electrical and Electronics Engineers Board of Directors in 2004. This award is presented for outstanding contributions to the advancement of the design, practice, techniques, or theory in biologically and linguistically motivated computational paradigms and systems, including neural networks, connectionist systems, evolutionary computation, fuzzy systems, and hybrid intelligent systems in which these paradigms are contained.\n\n"
            "The award may be presented to an individual, multiple recipients, or a team of up to three people. It is named for Frank Rosenblatt, creator of the perceptron.\n\n"
            "Recipients of this award receive a bronze medal, certificate, and honorarium.\n\n"
            "Recipients\n"
            "- 2026: Andrew G. Barto & Richard S. Sutton\n"
            "- 2025: Yaochu Jin\n"
            "- 2024: Bernadette Bouchon-Meunier\n"
            "- 2023: Marios Polycarpou\n"
            "- 2022: Paul Werbos\n"
            "- 2021: James M. Keller\n"
            "- 2020: Xin Yao\n"
            "- 2019: Erkki Oja\n"
            "- 2018: Enrique H. Ruspini\n"
            "- 2017: Stephen Grossberg\n"
            "- 2016: Ronald R. Yager\n"
            "- 2015: Marco Dorigo\n"
            "- 2014: Geoffrey E. Hinton\n"
            "- 2013: Terrence Sejnowski\n"
            "- 2012: Vladimir Vapnik\n"
            "- 2011: Hans-Paul Schwefel\n"
            "- 2010: Michio Sugeno\n"
            "- 2009: John J. Hopfield\n"
            "- 2008: Teuvo Kohonen\n"
            "- 2007: James C. Bezdek\n"
            "- 2006: Lawrence J. Fogel\n\n"
            "[Original Question]\n"
            "Who received the IEEE Frank Rosenblatt Award in 2010?\n\n"
            "[Original Answer]\n"
            "Michio Sugeno\n\n"
            "[Rich Question]\n"
            "List all the IEEE Frank Rosenblatt Award recipients from 2010 to 2026.\n\n"
            "[Rich Answer]\n"
            "- 2026: Andrew G. Barto & Richard S. Sutton\n"
            "- 2025: Yaochu Jin\n"
            "- 2024: Bernadette Bouchon-Meunier\n"
            "- 2023: Marios Polycarpou\n"
            "- 2022: Paul Werbos\n"
            "- 2021: James M. Keller\n"
            "- 2020: Xin Yao\n"
            "- 2019: Erkki Oja\n"
            "- 2018: Enrique H. Ruspini\n"
            "- 2017: Stephen Grossberg\n"
            "- 2016: Ronald R. Yager\n"
            "- 2015: Marco Dorigo\n"
            "- 2014: Geoffrey E. Hinton\n"
            "- 2013: Terrence Sejnowski\n"
            "- 2012: Vladimir Vapnik\n"
            "- 2011: Hans-Paul Schwefel\n"
            "- 2010: Michio Sugeno\n\n"
            "### EXAMPLE 2 ###\n"
            "[Wikipedia Passage]\n"
            "Mirza Hameedullah Beg (M. H. Beg) (22 February 1913 – 19 November 1988) was the 15th Chief Justice of India, serving from January 1977 to February 1978. Appointed by Fakhruddin Ali Ahmed.\n\n"
            "[Original Question]\n"
            "Who appointed the Chief Justice of India, Mirza Hameedullah Beg, in 1977?\n\n"
            "[Original Answer]\n"
            "Fakhruddin Ali Ahmed\n\n"
            "[Rich Question]\n"
            "Provide the full name, lifespan, position, and term of service of the individual who appointed Mirza Hameedullah Beg as Chief Justice of India in 1977.\n\n"
            "[Rich Answer]\n"
            "The individual who appointed Mirza Hameedullah Beg as Chief Justice of India in 1977 was Fakhruddin Ali Ahmed (13 May 1905 – 11 February 1977), who served as the 5th President of India from 1974 until his death in 1977.\n\n"
            "### YOUR TASK ###\n"
            "[Wikipedia Passage]\n"
            f"{passage}\n\n"
            "[Original Question]\n"
            f"{question}\n\n"
            "[Original Answer]\n"
            f"{answer}\n\n"
            "Return JSON with keys rich_question and rich_answer only."
        )
        raw = self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            seed=seed,
        )
        self._log("[rich_qa] system_prompt=" + self._truncate(system_prompt))
        self._log("[rich_qa] user_prompt=" + self._truncate(user_prompt))
        self._log("[rich_qa] raw_response=" + self._truncate(raw))
        payload = parse_json_payload(raw)
        if not payload:
            return None
        rich_question = payload.get("rich_question")
        rich_answer = payload.get("rich_answer")
        if not isinstance(rich_question, str) or not rich_question.strip():
            return None
        if not isinstance(rich_answer, str) or not rich_answer.strip():
            return None
        return {"rich_question": rich_question.strip(), "rich_answer": rich_answer.strip()}

    def _verify_rich_qa(
        self,
        *,
        passage: str,
        original_question: str,
        original_answer: str,
        rich_question: str,
        rich_answer: str,
        seed: int | None,
    ) -> tuple[bool, str]:
        system_prompt = (
            "You are a verifier. Return JSON only with keys 'status' and 'reason'. "
            "Status must be 'pass' only when ALL of the following hold:\n"
            "1. COVERAGE: a careful reader can deduce the original answer from the rich Q/A — "
            "the rich answer does not need to restate the answer verbatim, but must contain it or clearly supporting context.\n"
            "2. BREADTH: the rich question requires knowing substantially more facts than the original. "
            "It FAILS breadth only when it is the original question with one or two minor extra clauses appended "
            "(e.g. '...and what does the award recognize?' or '...and who was the runner-up?'). "
            "It PASSES breadth if it asks for a meaningfully wider set of facts — even if topically related — "
            "such as a career overview, a list of multiple items, historical context, or a different framing "
            "that happens to include the original answer as one fact among several.\n"
            "Return 'fail' if either condition is violated, and state which condition failed and why."
        )
        user_prompt = (
            "Assess the rich Q/A pair against two criteria:\n"
            "1. COVERAGE: a careful reader can deduce the original answer from the rich Q/A. "
            "The rich answer does not need to state the answer verbatim — it passes if the answer is clearly implied or inferable. "
            "Only fail if the original answer is genuinely absent from the rich Q/A.\n"
            "2. BREADTH: the rich question requires knowing substantially more facts than the original. "
            "FAIL only if the rich question is the original with one or two minor extra clauses appended. "
            "PASS if the rich question asks for a career overview, a list of facts, historical context, "
            "or a differently-framed question that covers the original answer as one fact among several — "
            "even if it is topically related to the original question.\n"
            "Say 'pass' only if BOTH criteria are met. Otherwise say 'fail'.\n\n"
            "### EXAMPLE ###\n"
            "[Passage]\n"
            "The IEEE Frank Rosenblatt Award is a Technical Field Award established by the Institute of Electrical and Electronics Engineers Board of Directors in 2004. This award is presented for outstanding contributions to the advancement of the design, practice, techniques, or theory in biologically and linguistically motivated computational paradigms and systems, including neural networks, connectionist systems, evolutionary computation, fuzzy systems, and hybrid intelligent systems in which these paradigms are contained.\n\n"
            "Recipients\n"
            "- 2026: Andrew G. Barto & Richard S. Sutton\n"
            "- 2025: Yaochu Jin\n"
            "- 2024: Bernadette Bouchon-Meunier\n"
            "- 2023: Marios Polycarpou\n"
            "- 2022: Paul Werbos\n"
            "- 2021: James M. Keller\n"
            "- 2020: Xin Yao\n"
            "- 2019: Erkki Oja\n"
            "- 2018: Enrique H. Ruspini\n"
            "- 2017: Stephen Grossberg\n"
            "- 2016: Ronald R. Yager\n"
            "- 2015: Marco Dorigo\n"
            "- 2014: Geoffrey E. Hinton\n"
            "- 2013: Terrence Sejnowski\n"
            "- 2012: Vladimir Vapnik\n"
            "- 2011: Hans-Paul Schwefel\n"
            "- 2010: Michio Sugeno\n\n"
            "[Original Question]\n"
            "Who received the IEEE Frank Rosenblatt Award in 2010?\n\n"
            "[Original Answer]\n"
            "Michio Sugeno\n\n"
            "[Rich Question]\n"
            "List all the IEEE Frank Rosenblatt Award recipients from 2010 to 2026.\n\n"
            "[Rich Answer]\n"
            "- 2026: Andrew G. Barto & Richard S. Sutton\n"
            "- 2025: Yaochu Jin\n"
            "- 2024: Bernadette Bouchon-Meunier\n"
            "- 2023: Marios Polycarpou\n"
            "- 2022: Paul Werbos\n"
            "- 2021: James M. Keller\n"
            "- 2020: Xin Yao\n"
            "- 2019: Erkki Oja\n"
            "- 2018: Enrique H. Ruspini\n"
            "- 2017: Stephen Grossberg\n"
            "- 2016: Ronald R. Yager\n"
            "- 2015: Marco Dorigo\n"
            "- 2014: Geoffrey E. Hinton\n"
            "- 2013: Terrence Sejnowski\n"
            "- 2012: Vladimir Vapnik\n"
            "- 2011: Hans-Paul Schwefel\n"
            "- 2010: Michio Sugeno\n\n"
            "Status: pass\n"
            "Reason: COVERAGE — the rich answer explicitly lists the 2010 recipient Michio Sugeno. "
            "BREADTH — the rich question asks for all recipients across a multi-year range, not just 2010.\n\n"
            "### COUNTER-EXAMPLE (fail — breadth) ###\n"
            "[Original Question]\n"
            "Which team won the Coppa Italia Serie C in the 1981-82 season?\n\n"
            "[Original Answer]\n"
            "Vigor Lamezia\n\n"
            "[Rich Question]\n"
            "Which team won the Coppa Italia Serie C in the 1981-82 season, and who was the runner-up?\n\n"
            "[Rich Answer]\n"
            "Vigor Lamezia won; the runner-up was Cavese.\n\n"
            "Status: fail\n"
            "Reason: BREADTH — the rich question is the original with only 'and who was the runner-up?' appended. "
            "That is one minor extra clause, not a broader question.\n\n"
            "### COUNTER-EXAMPLE (pass — breadth, even though topically related) ###\n"
            "[Original Question]\n"
            "In what year was Alain Stanké made a member of the Order of Canada?\n\n"
            "[Original Answer]\n"
            "1998\n\n"
            "[Rich Question]\n"
            "What major honors and awards did Alain Stanké receive throughout his career, and when?\n\n"
            "[Rich Answer]\n"
            "Alain Stanké received the Order of Canada in 1998, the Prix du Québec in 2003, and the Governor General's Award in 2007.\n\n"
            "Status: pass\n"
            "Reason: COVERAGE — the rich answer includes the 1998 Order of Canada, so the original answer is deducible. "
            "BREADTH — the rich question asks for multiple honors across a career, requiring substantially more facts than just one year.\n\n"
            "### YOUR TASK ###\n"
            "[Passage]\n"
            f"{passage}\n\n"
            "[Original Question]\n"
            f"{original_question}\n\n"
            "[Original Answer]\n"
            f"{original_answer}\n\n"
            "[Rich Question]\n"
            f"{rich_question}\n\n"
            "[Rich Answer]\n"
            f"{rich_answer}\n\n"
            "Return JSON with keys status and reason only."
        )
        raw = self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            seed=seed,
        )
        self._log("[verify] system_prompt=" + self._truncate(system_prompt))
        self._log("[verify] user_prompt=" + self._truncate(user_prompt))
        self._log("[verify] raw_response=" + self._truncate(raw))

        payload = parse_json_payload(raw)
        if not payload:
            return False, "could not parse verification response"
        status = payload.get("status")
        reason = payload.get("reason")
        reason_text = reason if isinstance(reason, str) else ""
        if status == "pass":
            return True, reason_text or "verified"
        fail_reason = reason_text or "verification status is not pass"
        return False, fail_reason

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> List[dict]:
        self._inc("total_samples")
        self._logged_samples += 1
        self._log("=" * 40)
        self._log(f"[sample] seed={rng_seed}")

        question = sample.get("question")
        if not isinstance(question, str) or not question.strip():
            question = get_prompt_text(sample)
        question_text = question.strip() if isinstance(question, str) else ""

        answer = sample.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            reward_model = sample.get("reward_model")
            if isinstance(reward_model, dict):
                answer = reward_model.get("ground_truth")
        answer_text = answer.strip() if isinstance(answer, str) else ""

        urls = extract_urls(sample)
        if not urls:
            self._log("[sample] no urls")
            self._log_failure("skipped_no_urls", sample, question=question_text, details="no urls fetched")
            self._inc("skipped_no_urls")
            return []
        self._log("[sample] urls=" + ", ".join(urls))

        if not answer_text:
            self._log("[sample] missing answer")
            self._log_failure(
                "skipped_missing_answer",
                sample,
                question=question_text,
                details="no answer text or reward_model ground truth",
            )
            self._inc("skipped_missing_answer")
            return []

        window: Optional[str] = None
        source_url: Optional[str] = None
        any_content = False
        for content, url in iter_source_texts(urls, self._timeout, self._max_chars):
            any_content = True
            self._log(f"[sample] trying source_url={url}")
            self._log("[sample] content_preview=" + self._truncate(content))

            # Local search first — no LLM call needed if answer is found directly
            local_window = find_answer_window(content, answer_text)
            if local_window:
                self._log(f"[sample] local search found window in url={url}")
                window = local_window
                source_url = url
                break

            # Fall back to LLM extraction on truncated content
            try:
                w = self._extract_window(
                    content=content[: self._max_chars],
                    question=question_text,
                    answer=answer_text,
                    seed=rng_seed,
                )
            except Exception as exc:
                self._log(f"[sample] extract window exception for url={url}: {exc}")
                continue
            if w:
                window = w
                source_url = url
                break

        if not any_content:
            self._log("[sample] no content fetched from any url")
            self._log_failure(
                "skipped_no_content",
                sample,
                question=question_text,
                details=f"no usable content from any url (timeout={self._timeout}, max_chars={self._max_chars})",
            )
            self._inc("skipped_no_content")
            return []

        if not window:
            self._log("[sample] no window found across all urls")
            self._log_failure(
                "skipped_no_window",
                sample,
                question=question_text,
                answer=answer_text,
                details="no url yielded a window containing the answer",
            )
            self._inc("skipped_no_window")
            return []
        self._log("[sample] window_preview=" + self._truncate(window))

        try:
            rich = self._generate_rich_qa(
                passage=window,
                question=question_text,
                answer=answer_text,
                seed=rng_seed,
            )
        except Exception as exc:  # pragma: no cover - best-effort guard against unexpected failures
            self._log("[sample] rich qa generation exception")
            self._log_failure(
                "skipped_rich_failed",
                sample,
                question=question_text,
                answer=answer_text,
                details=f"exception during generation: {exc}",
            )
            self._inc("skipped_rich_failed")
            return []
        if not rich:
            self._log("[sample] rich qa generation failed")
            self._log_failure(
                "skipped_rich_failed",
                sample,
                question=question_text,
                answer=answer_text,
                details="rich generation returned no payload",
            )
            self._inc("skipped_rich_failed")
            return []

        rich_question = rich["rich_question"]
        rich_answer = rich["rich_answer"]

        try:
            verified, verification_reason = self._verify_rich_qa(
                passage=window,
                original_question=question_text,
                original_answer=answer_text,
                rich_question=rich_question,
                rich_answer=rich_answer,
                seed=rng_seed,
            )
        except Exception as exc:  # pragma: no cover - best-effort guard against unexpected failures
            self._log("[sample] verification exception")
            self._log_failure(
                "skipped_verification_failed",
                sample,
                question=question_text,
                answer=answer_text,
                details=f"exception during verification: {exc}",
            )
            self._inc("skipped_verification_failed")
            return []
        self._log(
            f"[sample] verification={'pass' if verified else 'fail'} reason={verification_reason}"
        )
        if not verified:
            self._log_failure(
                "skipped_verification_failed",
                sample,
                question=question_text,
                answer=answer_text,
                details=verification_reason,
            )
            self._inc("skipped_verification_failed")
            return []

        updated = deepcopy(sample)

        updated["prompt"] = [{"role": "user", "content": rich_question}]
        if "question" in updated:
            updated["question"] = rich_question
        if "answer" in updated:
            updated["answer"] = rich_answer
        extra_info = updated.get("extra_info")
        if isinstance(extra_info, dict) and "question" in extra_info:
            extra_info["original_question"] = extra_info.get("question")
            extra_info["original_answer"] = extra_info.get("answer")
        reward_model = updated.get("reward_model")
        if isinstance(reward_model, dict) and "ground_truth" in reward_model:
            reward_model["ground_truth"] = rich_answer

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
