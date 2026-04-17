"""Generate a new rich SimpleQA parquet with k paraphrased QA pairs per row."""

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


DEFAULT_INPUT = "/work/hdd/bbsg/echen10/data/simpleqa/augment/rich_sft/simpleqa_rich_sft_train.parquet"
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


def _extract_answer(sample: dict[str, Any]) -> str:
    answer = _strip_text(sample.get("answer"))
    if answer:
        return answer
    reward_model = sample.get("reward_model")
    if isinstance(reward_model, dict):
        return _strip_text(reward_model.get("ground_truth"))
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


@register("simpleqa_n_rich_sft")
class SimpleqaNRichSftRewriter:
    name = "simpleqa_n_rich_sft"

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

    def _rewrite_qa(self, *, question: str, answer: str, seed: int | None) -> Optional[dict[str, str]]:
        system_prompt = (
            "You generate paraphrases of factual QA pairs.\n"
            "Preserve the exact meaning and exact answer.\n"
            "Do not change entities, numbers, dates, or qualifiers.\n"
            "Do not introduce new facts or ambiguity.\n"
            "Return JSON only with keys 'question' and 'answer'."
        )
        user_prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Rewrite the QA pair with different wording while keeping the same fact.\n\n"
            "Rules:\n"
            "- The rewritten question must ask for the exact same information.\n"
            "- The answer must remain exactly the same string.\n"
            "- Preserve all qualifiers (time, location, superlatives, roles, etc.).\n"
            "- Do not introduce new facts or remove necessary information.\n"
            "- Avoid trivial edits; make the wording meaningfully different.\n\n"
            "Return JSON:\n"
            "{ \"question\": \"...\", \"answer\": \"...\" }"
        )

        raw = self._client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=self._temperature,
            seed=seed,
        )

        payload = _parse_json_payload(raw)
        if not payload:
            return None

        rewritten_question = _strip_text(payload.get("question"))
        rewritten_answer = _strip_text(payload.get("answer"))
        if not rewritten_question or not rewritten_answer:
            return None

        return {
            "question": rewritten_question,
            "answer": rewritten_answer,
        }

    def rewrite(self, sample: dict, *, rng_seed: int | None = None) -> list[dict]:
        question = _extract_question(sample)
        answer = _extract_answer(sample)
        if not question or not answer:
            return []

        rewritten = None
        for attempt_idx in range(self._max_retries):
            seed = None if rng_seed is None else _derive_seed(rng_seed, "single", 0, attempt_idx)
            rewritten = self._rewrite_qa(question=question, answer=answer, seed=seed)
            if rewritten is not None:
                break

        if rewritten is None:
            return []

        updated = deepcopy(sample)
        rewritten_question = rewritten["question"]
        rewritten_answer = rewritten["answer"]

        _set_prompt_question(updated, rewritten_question)

        if "question" in updated:
            updated["question"] = rewritten_question
        if "answer" in updated:
            updated["answer"] = rewritten_answer

        extra_info = updated.get("extra_info")
        if isinstance(extra_info, dict):
            if "question" in extra_info:
                extra_info["original_question"] = extra_info.get("question")
                extra_info["question"] = rewritten_question
            if "answer" in extra_info:
                extra_info["original_answer"] = extra_info.get("answer")
                extra_info["answer"] = rewritten_answer

        reward_model = updated.get("reward_model")
        if isinstance(reward_model, dict) and "ground_truth" in reward_model:
            reward_model["ground_truth"] = rewritten_answer

        updated = attach_augmentation_metadata(
            updated,
            method_name=self.name,
            variant_idx=0,
            params={
                "model": self._client.model,
                "temperature": self._temperature,
            },
            seed=rng_seed,
        )
        return [updated]


def build_augmented_parquet(
    *,
    input_path: str,
    output_path: str,
    k: int,
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

    if k < 1:
        raise ValueError("k must be >= 1")
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be >= 0")

    df = pd.read_parquet(input_path)
    if "question" not in df.columns and "prompt" not in df.columns:
        raise ValueError("Input parquet must contain either 'question' or 'prompt' column")

    total_rows = len(df)
    if max_samples is not None:
        total_rows = min(total_rows, max_samples)

    rewriter = SimpleqaNRichSftRewriter(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
    )

    output_rows: list[dict[str, Any]] = []
    success_count = 0
    fallback_count = 0
    start_row_idx = 0

    if resume and os.path.exists(checkpoint_state_path) and os.path.exists(checkpoint_path):
        with open(checkpoint_state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        expected = {
            "input_path": os.path.abspath(input_path),
            "k": k,
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
                "k": k,
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

    for row_idx in range(start_row_idx, total_rows):
        sample = df.iloc[row_idx].to_dict()
        sample_id = str(row_idx)
        extra_info = sample.get("extra_info")
        if isinstance(extra_info, dict):
            extra_sample_id = extra_info.get("sample_id")
            if extra_sample_id is not None:
                sample_id = str(extra_sample_id)

        for variant_idx in range(k):
            derived_seed = _derive_seed(seed, sample_id, variant_idx, 0)
            rewritten: list[dict[str, Any]] = []
            rewrite_error: str | None = None
            try:
                rewritten = rewriter.rewrite(sample, rng_seed=derived_seed)
            except Exception as exc:
                rewrite_error = _truncate(f"{type(exc).__name__}: {exc}")
                print(
                    f"[warn] row={row_idx} variant={variant_idx} rewrite failed; "
                    f"using fallback. error={rewrite_error}"
                )
            if rewritten:
                rewritten_row = attach_augmentation_metadata(
                    rewritten[0],
                    method_name=rewriter.name,
                    variant_idx=variant_idx,
                    params={
                        "model": model,
                        "temperature": temperature,
                    },
                    seed=derived_seed,
                )
                output_rows.append(rewritten_row)
                success_count += 1
                continue

            fallback_count += 1
            if drop_failed:
                continue

            fallback = deepcopy(sample)
            fallback = attach_augmentation_metadata(
                fallback,
                method_name=rewriter.name,
                variant_idx=variant_idx,
                params={
                    "model": model,
                    "temperature": temperature,
                    "fallback": "rewrite_failed" if rewrite_error else "kept_original",
                    "error": rewrite_error,
                },
                seed=derived_seed,
            )
            output_rows.append(fallback)

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
        "k": k,
        "written_rows": len(out_df),
        "successful_rewrites": success_count,
        "fallback_rows": fallback_count,
    }


def _default_output_path(input_path: str, k: int) -> str:
    path = Path(input_path)
    return str(path.with_name(f"{path.stem}_k{k}{path.suffix}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new rich SimpleQA parquet where each source QA generates k "
            "paraphrased QA rows."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input rich SimpleQA parquet",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output parquet path (default: <input>_k{K}.parquet)",
    )
    parser.add_argument("--k", type=int, required=True, help="Number of rewrites per input QA")
    parser.add_argument("--model", default="gpt-5-nano", help="OpenAI model")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Global seed")
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum rewrite attempts per generated QA",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for debug runs")
    parser.add_argument(
        "--drop_failed",
        action="store_true",
        help="Drop failed rewrites instead of writing fallback original rows",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help=(
            "Save an on-disk checkpoint every N source rows. "
            "Set to 0 to disable checkpointing."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint parquet path (default: <output>.checkpoint.parquet)",
    )
    parser.add_argument(
        "--checkpoint_state_path",
        default=None,
        help="Checkpoint state JSON path (default: <checkpoint>.state.json)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint files if available.",
    )
    parser.add_argument(
        "--keep_checkpoint",
        action="store_true",
        help="Keep checkpoint files after successful completion.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = args.output or _default_output_path(args.input, args.k)
    checkpoint_path = args.checkpoint_path or _default_checkpoint_path(output_path)
    checkpoint_state_path = args.checkpoint_state_path or _default_checkpoint_state_path(checkpoint_path)

    stats = build_augmented_parquet(
        input_path=args.input,
        output_path=output_path,
        k=args.k,
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
        f"k={stats['k']}, "
        f"written_rows={stats['written_rows']}, "
        f"successful_rewrites={stats['successful_rewrites']}, "
        f"fallback_rows={stats['fallback_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
