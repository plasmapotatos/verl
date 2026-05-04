"""Inverse-question rewriter.

Given a (question, answer) pair, produce an inverse question where the original
answer becomes implicit context and a different entity from the original
question becomes the new answer target.

Example:
  original: "What month and year did Obama tell Christianity Today he was a
            devout Christian?" -> answer: "January 2008"
  inverse:  "To what publication did Obama declare he was a devout Christian
            in January 2008?" -> answer: "Christianity Today"
"""

from __future__ import annotations

from copy import deepcopy
from typing import List, Optional

from ..openai_client import OpenAIClient
from ..registry import register
from ..schemas import attach_augmentation_metadata, get_prompt_text
from .simpleqa_web_utils import parse_json_payload


@register("inverse_qa")
class InverseQaRewriter:
    name = "inverse_qa"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        timeout: int = 60,
    ) -> None:
        self._client = OpenAIClient(model, timeout=float(timeout))
        self._metrics = {
            "total_samples": 0,
            "skipped_missing_answer": 0,
            "skipped_generation_failed": 0,
            "success": 0,
        }

    def get_metrics(self) -> dict:
        return dict(self._metrics)

    def _inc(self, key: str) -> None:
        if key in self._metrics:
            self._metrics[key] += 1

    def _generate_inverse(
        self,
        *,
        question: str,
        answer: str,
        seed: int | None,
    ) -> Optional[dict]:
        system_prompt = (
            "You rewrite question/answer pairs into INVERSE pairs. "
            "Given an original (question, answer), choose a salient entity inside the "
            "original question that is not the original answer, and produce a new "
            "question that asks for that entity. The original answer must be folded "
            "into the new question as implicit context (a known fact). "
            "Return JSON only with keys 'inverse_question' and 'inverse_answer'."
        )
        user_prompt = (
            "Rewrite the (question, answer) pair as an inverse pair.\n\n"
            "Rules:\n"
            "1. The new answer must be a SHORT entity (name, place, publication, "
            "organization, title, etc.) lifted directly from the original question — "
            "not the original answer.\n"
            "2. The new question must contain the original answer as a known fact.\n"
            "3. Together, the new (question, answer) must encode the same underlying "
            "fact as the original.\n"
            "4. The new answer must be unambiguously the unique correct response to "
            "the new question given common world knowledge plus the implicit context.\n\n"
            "### EXAMPLE ###\n"
            "[Original Question]\n"
            "What month and year did Obama tell Christianity Today he was a devout Christian?\n"
            "[Original Answer]\n"
            "January 2008\n"
            "[Inverse Question]\n"
            "To what publication did Obama declare he was a devout Christian in January 2008?\n"
            "[Inverse Answer]\n"
            "Christianity Today\n\n"
            "### YOUR TASK ###\n"
            f"[Original Question]\n{question}\n"
            f"[Original Answer]\n{answer}\n\n"
            "Return JSON with keys inverse_question and inverse_answer only."
        )
        raw = self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            seed=seed,
        )
        payload = parse_json_payload(raw)
        if not payload:
            return None
        inverse_question = payload.get("inverse_question")
        inverse_answer = payload.get("inverse_answer")
        if not isinstance(inverse_question, str) or not inverse_question.strip():
            return None
        if not isinstance(inverse_answer, str) or not inverse_answer.strip():
            return None
        return {
            "inverse_question": inverse_question.strip(),
            "inverse_answer": inverse_answer.strip(),
        }

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> List[dict]:
        self._inc("total_samples")

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

        if not question_text or not answer_text:
            self._inc("skipped_missing_answer")
            return []

        try:
            inv = self._generate_inverse(
                question=question_text,
                answer=answer_text,
                seed=rng_seed,
            )
        except Exception:
            self._inc("skipped_generation_failed")
            return []
        if not inv:
            self._inc("skipped_generation_failed")
            return []

        inv_q = inv["inverse_question"]
        inv_a = inv["inverse_answer"]

        updated = deepcopy(sample)
        updated["prompt"] = [{"role": "user", "content": inv_q}]
        if "question" in updated:
            updated["question"] = inv_q
        if "answer" in updated:
            updated["answer"] = inv_a
        extra_info = updated.get("extra_info")
        if isinstance(extra_info, dict):
            if "question" in extra_info:
                extra_info["original_question"] = extra_info.get("question")
                extra_info["question"] = inv_q
            if "answer" in extra_info:
                extra_info["original_answer"] = extra_info.get("answer")
                extra_info["answer"] = inv_a
        reward_model = updated.get("reward_model")
        if isinstance(reward_model, dict) and "ground_truth" in reward_model:
            reward_model["ground_truth"] = inv_a

        updated = attach_augmentation_metadata(
            updated,
            method_name=self.name,
            variant_idx=0,
            params={"model": self._client.model},
            seed=rng_seed,
        )
        self._inc("success")
        return [updated]
