"""Generate a new SimpleQA parquet with n total QA rows per input (1 original + n-1 paraphrased questions).

The short answer is kept exactly; only the question string is paraphrased. The CoT `answer` field
(which may reference the question wording) is left untouched — downstream users should decide whether
to regenerate CoT if they require question/CoT consistency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

try:
    from ..openai_client import OpenAIClient
    from ..registry import register
    from ..schemas import attach_augmentation_metadata, get_prompt_text
except ImportError:  # pragma: no cover - allows direct script execution
    import sys

    package_root = Path(__file__).resolve().parents[3]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from verl.augment.openai_client import OpenAIClient
    from verl.augment.registry import register
    from verl.augment.schemas import attach_augmentation_metadata, get_prompt_text


DEFAULT_INPUT = "/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/partition/rich_qa/cot/sft/train.parquet"
DEFAULT_CHECKPOINT_EVERY = 100


def _parse_json_payload(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    left = text.find("{")
    right = text.rfind("}")
    if left < 0 or right <= left:
        return None
    try:
        payload = json.loads(text[left : right + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _strip_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_question(sample: dict[str, Any]) -> str:
    question = _strip_text(sample.get("question"))
    if question:
        return question
    try:
        return _strip_text(get_prompt_text(sample))
    except Exception:
        return ""


def _extract_short_answer(sample: dict[str, Any]) -> str:
    """Pull the short factual answer (not the CoT `answer` field)."""
    reward_model = sample.get("reward_model")
    if isinstance(reward_model, dict):
        gt = _strip_text(reward_model.get("ground_truth"))
        if gt:
            return gt
    extra_info = sample.get("extra_info")
    if isinstance(extra_info, dict):
        orig = _strip_text(extra_info.get("original_answer"))
        if orig:
            return orig
        ans = _strip_text(extra_info.get("answer"))
        if ans:
            return ans
    return ""


def _set_prompt_question(sample: dict[str, Any], question: str) -> None:
    prompt = sample.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        return
    first = prompt[0]
    if not isinstance(first, dict):
        return
    first["content"] = question


def _derive_seed(global_seed: int, sample_id: str, variant_idx: int, attempt_idx: int) -> int:
    payload = f"{global_seed}|{sample_id}|{variant_idx}|{attempt_idx}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _truncate(text: str, max_len: int = 400) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _default_checkpoint_path(output_path: str) -> str:
    path = Path(output_path)
    return str(path.with_name(f"{path.stem}.checkpoint{path.suffix}"))


def _default_checkpoint_state_path(checkpoint_path: str) -> str:
    path = Path(checkpoint_path)
    return str(path.with_name(f"{path.stem}.state.json"))


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".json.tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _write_parquet_atomic(df: Any, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=out_dir)
    os.close(fd)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _build_output_df(*, source_df: Any, output_rows: list[dict[str, Any]]) -> Any:
    import pandas as pd

    out_df = pd.DataFrame(output_rows)
    if out_df.empty:
        return source_df.iloc[0:0].copy()
    ordered_cols = list(source_df.columns) + [col for col in out_df.columns if col not in source_df.columns]
    return out_df[ordered_cols]


SYSTEM_PROMPT = (
    "You paraphrase short factual questions for a QA dataset. "
    "You aggressively vary syntactic structure across paraphrases. You return JSON only."
)


# Strategies are indexed; when n=1 per call (the default in our CLI), the rewriter
# derives one strategy per variant from rng_seed so sibling variants of the same
# source question land on different frames instead of all collapsing to the modal
# "to whom was X given" passive form.
PARAPHRASE_STRATEGIES: list[dict[str, str]] = [
    {
        "name": "passive_voice",
        "instruction": "Use passive voice; the original subject becomes the by-phrase or disappears.",
    },
    {
        "name": "cleft_or_fronting",
        "instruction": "Use a cleft or fronted construction (e.g. 'It was ___ that...', 'The ___ that ___ is what?').",
    },
    {
        "name": "embedded_clause",
        "instruction": "Embed the query inside an outer clause (e.g. 'Can you identify the ___?', 'Do you know which ___?', \"I'd like to know ___\").",
    },
    {
        "name": "reorder_context",
        "instruction": "Move the contextual modifier (year, place, event) to a different position than in the original.",
    },
    {
        "name": "nominalize_qword",
        "instruction": "Replace the question word with a nominal or different question word ('who'->'which person', 'when'->'in what year', 'what'->'name the ___').",
    },
    {
        "name": "imperative_name",
        "instruction": "Phrase as an imperative starting with 'Name ___', 'Identify ___', or 'State ___' rather than as a wh-question.",
    },
    {
        "name": "tail_question",
        "instruction": "Phrase as a near-declarative with a question tail (e.g. 'The ___ in ___ went to whom?').",
    },
]


# Few-shot showing FIVE maximally different surface forms for one Q/A, plus a
# negative example calling out the failure mode we observed -- every variant
# collapsing to 'to whom was X given' -- which the model is otherwise strongly
# biased toward.
FEW_SHOT_BLOCK = (
    "Example A\n"
    "  Question: Who received the IEEE Frank Rosenblatt Award in 2010?\n"
    "  Answer: Michio Sugeno\n"
    "  Five GOOD paraphrases (each uses a DIFFERENT structure):\n"
    "    1. (passive_voice)        The IEEE Frank Rosenblatt Award in 2010 was given to whom?\n"
    "    2. (cleft_or_fronting)    In 2010, it was which person that received the IEEE Frank Rosenblatt Award?\n"
    "    3. (embedded_clause)      Can you identify the recipient of the 2010 IEEE Frank Rosenblatt Award?\n"
    "    4. (reorder_context)      For 2010, which individual was honored with the IEEE Frank Rosenblatt Award?\n"
    "    5. (imperative_name)      Name the 2010 IEEE Frank Rosenblatt Award winner.\n\n"
    "Example B\n"
    "  Question: In which year did the Treaty of Versailles end World War I?\n"
    "  Answer: 1919\n"
    "  Five GOOD paraphrases (each uses a DIFFERENT structure):\n"
    "    1. (passive_voice)        World War I was ended by the Treaty of Versailles in what year?\n"
    "    2. (cleft_or_fronting)    It was in which year that the Treaty of Versailles concluded World War I?\n"
    "    3. (embedded_clause)      Do you know the year in which the Treaty of Versailles brought World War I to a close?\n"
    "    4. (tail_question)        The Treaty of Versailles ended World War I -- in what year?\n"
    "    5. (imperative_name)      State the year the Treaty of Versailles ended World War I.\n\n"
    "BAD example (do NOT do this -- all five collapse to the same passive 'to whom was X' frame):\n"
    "    1. To whom was the award given in 2010?\n"
    "    2. To whom was the award awarded in 2010?\n"
    "    3. To whom was the award presented in 2010?\n"
    "    4. To whom was the award granted in 2010?\n"
    "    5. To whom was the award handed in 2010?\n"
    "  ^ This is the failure mode. Synonym swaps over a single frame are NOT paraphrases.\n"
)


def _build_user_prompt(
    *,
    question: str,
    answer: str,
    n_paraphrases: int,
    forced_strategy: dict[str, str] | None = None,
) -> str:
    if forced_strategy is not None and n_paraphrases == 1:
        strategy_block = (
            "For your single paraphrase, you MUST use this specific strategy "
            "(do not silently fall back to a different one):\n"
            f"  Strategy: {forced_strategy['name']}\n"
            f"  Instruction: {forced_strategy['instruction']}\n\n"
        )
    else:
        strategy_block = (
            "Across the paraphrases, use a DIFFERENT strategy for each one. Available strategies:\n"
            + "\n".join(f"  - {s['name']}: {s['instruction']}" for s in PARAPHRASE_STRATEGIES)
            + "\n\n"
        )

    return (
        "You paraphrase short factual questions for a QA dataset.\n\n"
        f"Given a question and its answer, generate {n_paraphrases} paraphrase(s) that:\n"
        "  1. Are answered by EXACTLY the same answer -- do not change what is being asked for.\n"
        "  2. Vary the syntactic STRUCTURE substantially -- not just synonym substitution over the same frame.\n"
        "  3. Are natural, fluent English questions.\n\n"
        "Hard constraints:\n"
        "  - Never negate or invert the question.\n"
        "  - Never add facts not implied by the original.\n"
        "  - The answer string must remain valid for every paraphrase.\n"
        "  - Avoid the overused 'to whom was X given/awarded/presented' passive frame unless the chosen strategy is explicitly passive_voice; even then, prefer fresher passives.\n\n"
        f"{strategy_block}"
        "Few-shot examples:\n"
        f"{FEW_SHOT_BLOCK}\n"
        f"Output a JSON object with a single key \"paraphrases\" whose value is a list of {n_paraphrases} string(s). No other keys, no commentary.\n\n"
        "---\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Output:"
    )


@register("simpleqa_n_paraphrase")
class SimpleqaNParaphraseRewriter:
    name = "simpleqa_n_paraphrase"

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> None:
        self._client = OpenAIClient(model)
        self._temperature = temperature
        self._max_retries = max_retries

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> list[dict]:
        """Produce a single paraphrased variant — used by the shared augment runner.

        The augment CLI calls this once per (sample, variant_idx), so we ask the model for
        a single paraphrase per call and let the thread pool parallelize across samples.
        """
        question = _extract_question(sample)
        short_answer = _extract_short_answer(sample)
        if not question or not short_answer:
            return []
        paraphrases = self.paraphrase_questions(
            question=question,
            short_answer=short_answer,
            n_paraphrases=1,
            rng_seed=rng_seed,
        )
        if not paraphrases:
            return []
        updated = deepcopy(sample)
        _set_prompt_question(updated, paraphrases[0])
        if "question" in updated:
            updated["question"] = paraphrases[0]
        extra_info = updated.get("extra_info")
        if isinstance(extra_info, dict):
            if "original_question" not in extra_info:
                extra_info["original_question"] = question
            extra_info["question"] = paraphrases[0]
        return [updated]

    def paraphrase_questions(
        self,
        *,
        question: str,
        short_answer: str,
        n_paraphrases: int,
        rng_seed: int | None,
    ) -> list[str]:
        if n_paraphrases <= 0:
            return []
        # When the runner asks for one paraphrase at a time (the default), pin each
        # variant to a specific strategy via rng_seed so sibling variants of the same
        # source question diversify across structures rather than all collapsing onto
        # the modal passive form.
        forced_strategy = None
        if n_paraphrases == 1 and rng_seed is not None:
            forced_strategy = PARAPHRASE_STRATEGIES[rng_seed % len(PARAPHRASE_STRATEGIES)]
        for attempt_idx in range(self._max_retries):
            seed = None if rng_seed is None else _derive_seed(rng_seed, "batch", 0, attempt_idx)
            raw = self._client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(
                    question=question,
                    answer=short_answer,
                    n_paraphrases=n_paraphrases,
                    forced_strategy=forced_strategy,
                ),
                temperature=self._temperature,
                seed=seed,
            )
            payload = _parse_json_payload(raw)
            if not payload:
                continue
            raw_list = payload.get("paraphrases")
            if not isinstance(raw_list, list):
                continue
            paraphrases = [_strip_text(x) for x in raw_list if _strip_text(x)]
            if len(paraphrases) >= n_paraphrases:
                return paraphrases[:n_paraphrases]
        return []


def _make_variant_row(
    *,
    base_sample: dict[str, Any],
    new_question: Optional[str],
    method_name: str,
    variant_idx: int,
    params: dict[str, Any],
    seed: int | None,
) -> dict[str, Any]:
    updated = deepcopy(base_sample)
    if new_question is not None:
        _set_prompt_question(updated, new_question)
        if "question" in updated:
            updated["question"] = new_question
        extra_info = updated.get("extra_info")
        if isinstance(extra_info, dict):
            if "original_question" not in extra_info:
                extra_info["original_question"] = _extract_question(base_sample)
            extra_info["question"] = new_question
    return attach_augmentation_metadata(
        updated,
        method_name=method_name,
        variant_idx=variant_idx,
        params=params,
        seed=seed,
    )


def build_augmented_parquet(
    *,
    input_path: str,
    output_path: str,
    n: int,
    model: str,
    temperature: float,
    seed: int,
    max_retries: int,
    max_samples: int | None,
    drop_failed: bool,
    checkpoint_every: int,
    checkpoint_path: str,
    checkpoint_state_path: str,
    resume: bool,
    keep_checkpoint: bool,
) -> dict[str, int]:
    import pandas as pd

    if n < 1:
        raise ValueError("n must be >= 1")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0")

    df = pd.read_parquet(input_path)
    if "question" not in df.columns and "prompt" not in df.columns:
        raise ValueError("Input parquet must contain either 'question' or 'prompt' column")

    total_rows = len(df)
    if max_samples is not None:
        total_rows = min(total_rows, max_samples)

    rewriter = SimpleqaNParaphraseRewriter(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )

    method_name = rewriter.name
    params = {"model": model, "temperature": temperature, "n": n}

    output_rows: list[dict[str, Any]] = []
    success_count = 0
    fallback_count = 0
    start_row_idx = 0

    if resume and os.path.exists(checkpoint_state_path) and os.path.exists(checkpoint_path):
        with open(checkpoint_state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        expected = {
            "input_path": os.path.abspath(input_path),
            "n": n,
            "model": model,
            "temperature": temperature,
            "seed": seed,
            "max_retries": max_retries,
            "drop_failed": drop_failed,
            "max_samples": max_samples,
            "source_rows": total_rows,
        }
        mismatched = [key for key, value in expected.items() if state.get(key) != value]
        if mismatched:
            mismatch_preview = ", ".join(mismatched[:5])
            raise ValueError(
                f"Checkpoint does not match current settings for keys: {mismatch_preview}. "
                "Use a compatible checkpoint or start without --resume."
            )
        checkpoint_df = pd.read_parquet(checkpoint_path)
        output_rows = checkpoint_df.to_dict("records")
        success_count = int(state.get("successful_rewrites", 0))
        fallback_count = int(state.get("fallback_rows", 0))
        start_row_idx = int(state.get("next_row_idx", 0))
        start_row_idx = min(max(start_row_idx, 0), total_rows)
        print(
            "Resuming from checkpoint: "
            f"next_row={start_row_idx}/{total_rows}, "
            f"recovered_rows={len(output_rows)}, "
            f"checkpoint={checkpoint_path}"
        )
    elif resume:
        print(
            "Resume requested but checkpoint files were not found. "
            "Starting from row 0."
        )

    def save_checkpoint(next_row_idx: int) -> None:
        checkpoint_df = _build_output_df(source_df=df, output_rows=output_rows)
        _write_parquet_atomic(checkpoint_df, checkpoint_path)
        _write_json_atomic(
            checkpoint_state_path,
            {
                "input_path": os.path.abspath(input_path),
                "output_path": os.path.abspath(output_path),
                "checkpoint_path": os.path.abspath(checkpoint_path),
                "checkpoint_state_path": os.path.abspath(checkpoint_state_path),
                "source_rows": total_rows,
                "next_row_idx": next_row_idx,
                "n": n,
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "max_retries": max_retries,
                "drop_failed": drop_failed,
                "max_samples": max_samples,
                "written_rows": len(output_rows),
                "successful_rewrites": success_count,
                "fallback_rows": fallback_count,
            },
        )

    n_paraphrases = n - 1

    for row_idx in range(start_row_idx, total_rows):
        sample = df.iloc[row_idx].to_dict()
        sample_id = str(row_idx)
        extra_info = sample.get("extra_info")
        if isinstance(extra_info, dict):
            extra_sample_id = extra_info.get("sample_id")
            if extra_sample_id is not None:
                sample_id = str(extra_sample_id)

        question = _extract_question(sample)
        short_answer = _extract_short_answer(sample)

        # Variant 0 is always the original.
        output_rows.append(
            _make_variant_row(
                base_sample=sample,
                new_question=None,
                method_name=method_name,
                variant_idx=0,
                params={**params, "role": "original"},
                seed=_derive_seed(seed, sample_id, 0, 0),
            )
        )

        paraphrases: list[str] = []
        rewrite_error: str | None = None
        if n_paraphrases > 0 and question and short_answer:
            try:
                paraphrases = rewriter.paraphrase_questions(
                    question=question,
                    short_answer=short_answer,
                    n_paraphrases=n_paraphrases,
                    rng_seed=_derive_seed(seed, sample_id, 0, 0),
                )
            except Exception as exc:
                rewrite_error = _truncate(f"{type(exc).__name__}: {exc}")
                print(
                    f"[warn] row={row_idx} paraphrase batch failed; "
                    f"falling back. error={rewrite_error}"
                )

        for variant_idx in range(1, n):
            paraphrase = paraphrases[variant_idx - 1] if variant_idx - 1 < len(paraphrases) else None
            derived_seed = _derive_seed(seed, sample_id, variant_idx, 0)
            if paraphrase:
                output_rows.append(
                    _make_variant_row(
                        base_sample=sample,
                        new_question=paraphrase,
                        method_name=method_name,
                        variant_idx=variant_idx,
                        params={**params, "role": "paraphrase"},
                        seed=derived_seed,
                    )
                )
                success_count += 1
                continue

            fallback_count += 1
            if drop_failed:
                continue

            output_rows.append(
                _make_variant_row(
                    base_sample=sample,
                    new_question=None,
                    method_name=method_name,
                    variant_idx=variant_idx,
                    params={
                        **params,
                        "role": "fallback",
                        "fallback": "rewrite_failed" if rewrite_error else "kept_original",
                        "error": rewrite_error,
                    },
                    seed=derived_seed,
                )
            )

        if (row_idx + 1) % 100 == 0:
            print(f"Processed {row_idx + 1}/{total_rows} source rows")
        if checkpoint_every > 0 and (row_idx + 1) % checkpoint_every == 0:
            save_checkpoint(next_row_idx=row_idx + 1)
            print(
                f"Checkpoint saved at source row {row_idx + 1}/{total_rows} "
                f"to {checkpoint_path}"
            )

    out_df = _build_output_df(source_df=df, output_rows=output_rows)
    _write_parquet_atomic(out_df, output_path)

    if checkpoint_every > 0 and not keep_checkpoint:
        for path in (checkpoint_path, checkpoint_state_path):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    print(f"[warn] unable to remove checkpoint file {path}: {exc}")

    return {
        "source_rows": total_rows,
        "n": n,
        "written_rows": len(out_df),
        "successful_rewrites": success_count,
        "fallback_rows": fallback_count,
    }


def _default_output_path(input_path: str, n: int) -> str:
    path = Path(input_path)
    return str(path.with_name(f"{path.stem}_n{n}_paraphrase{path.suffix}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new SimpleQA parquet where each source QA is expanded into n rows "
            "(1 original + n-1 question paraphrases, sharing the same answer)."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input SimpleQA parquet")
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path (default: <input>_n{N}_paraphrase.parquet)",
    )
    parser.add_argument("--n", type=int, required=True, help="Total rows per input (1 original + n-1 paraphrases)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Global seed")
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum paraphrase attempts per source row (batched call)",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for debug runs")
    parser.add_argument(
        "--drop_failed",
        action="store_true",
        help="Drop failed paraphrase slots instead of writing fallback original rows",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Save an on-disk checkpoint every N source rows. Set to 0 to disable.",
    )
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--checkpoint_state_path", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument(
        "--keep_checkpoint",
        action="store_true",
        help="Keep checkpoint files after successful completion.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = args.output or _default_output_path(args.input, args.n)
    checkpoint_path = args.checkpoint_path or _default_checkpoint_path(output_path)
    checkpoint_state_path = args.checkpoint_state_path or _default_checkpoint_state_path(checkpoint_path)

    stats = build_augmented_parquet(
        input_path=args.input,
        output_path=output_path,
        n=args.n,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_retries=args.max_retries,
        max_samples=args.max_samples,
        drop_failed=args.drop_failed,
        checkpoint_every=args.checkpoint_every,
        checkpoint_path=checkpoint_path,
        checkpoint_state_path=checkpoint_state_path,
        resume=args.resume,
        keep_checkpoint=args.keep_checkpoint,
    )

    print(f"Input: {args.input}")
    print(f"Output: {output_path}")
    if args.checkpoint_every > 0:
        print(f"Checkpoint parquet: {checkpoint_path}")
        print(f"Checkpoint state: {checkpoint_state_path}")
    print(
        "Summary: "
        f"source_rows={stats['source_rows']}, "
        f"n={stats['n']}, "
        f"written_rows={stats['written_rows']}, "
        f"successful_rewrites={stats['successful_rewrites']}, "
        f"fallback_rows={stats['fallback_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
