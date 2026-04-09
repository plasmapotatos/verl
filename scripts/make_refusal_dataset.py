#!/usr/bin/env python3
"""Generate refusal-augmented SFT and RL datasets from an input parquet.

Pipeline:
  1. Split the input into an "answer" portion and a "refusal source" portion
     via scripts/make_eval_subset.py (fraction = --refusal-fraction).
  2. Rewrite the refusal-source portion with `verl.augment.cli` using
     method=simpleqa_refusal to obtain refusal targets.

  SFT outputs (written to <output-dir>/sft/):
  3. train.parquet          — combined answer + refusal
  4. train_origqa.parquet   — origqa combined (refusal remapped with
                              --preserve-answer-reward)
  5. Stratified eval splits (eval-fraction of each side):
       train_eval_answer.parquet, train_eval_refusal.parquet,
       train_eval_origqa_answer.parquet, train_eval_origqa_refusal.parquet
     Refusal eval files get a `target` column with the original factual
     answer so evaluation scores factual accuracy.

  RL outputs (written to <output-dir>/rl/):
  6. 80/20 split of each origqa side (answer_origqa, refusal_origqa).
  7. train.parquet           — combine the 80% portions
  8. train_eval_answer.parquet, train_eval_refusal.parquet
                             — eval-fraction sample from each 80% portion
  9. val_answer.parquet, val_refusal.parquet
                             — eval-fraction sample from each 20% portion
     Refusal eval/val files also get the factual `target` column.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent


def run(cmd: list[str]) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(REPO_ROOT))


def make_subset(
    input_path: Path,
    out_path: Path,
    fraction: float,
    seed: int,
    save_remainder: bool,
    train_output: Path | None,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "make_eval_subset.py"),
        "--input", str(input_path),
        "--output", str(out_path),
        "--fraction", f"{fraction}",
        "--seed", str(seed),
    ]
    if save_remainder:
        cmd.append("--save-remainder")
        if train_output is not None:
            cmd += ["--train-output", str(train_output)]
    run(cmd)


def augment_refusal(input_path: Path, output_path: Path, seed: int) -> None:
    cmd = [
        sys.executable, "-m", "verl.augment.cli",
        "--input", str(input_path),
        "--output", str(output_path),
        "--method", "simpleqa_refusal",
        "--seed", str(seed),
    ]
    run(cmd)


def combine(paths: list[Path], out_dir: Path, shuffle: bool, seed: int) -> Path:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "combine_parquet.py"),
        "--input_paths", *[str(p) for p in paths],
        "--output_dir", str(out_dir),
        "--seed", str(seed),
    ]
    if not shuffle:
        cmd.append("--no-shuffle")
    run(cmd)
    return out_dir / "combined.parquet"


def fill_target(primary: Path, reference: Path, output: Path) -> None:
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "fill_target_from_ref.parquet.py"),
        "--primary", str(primary),
        "--reference", str(reference),
        "--output", str(output),
    ]
    run(cmd)


def remap(
    input_path: Path,
    output_path: Path,
    base: Path | None,
    preserve_answer_reward: bool,
) -> None:
    cmd = [
        sys.executable, str(SCRIPTS_DIR / "remap_fields_by_id.py"),
        "--input", str(input_path),
        "--output", str(output_path),
    ]
    if base is not None:
        cmd += ["--base", str(base)]
    if preserve_answer_reward:
        cmd.append("--preserve-answer-reward")
    run(cmd)


def _copy(src: Path, dst: Path, label: str) -> None:
    shutil.copyfile(src, dst)
    print(f"Wrote {label} -> {dst}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input parquet path")
    ap.add_argument(
        "--refusal-fraction",
        type=float,
        default=0.5,
        help="Fraction of rows to convert into refusal targets (default 0.5)",
    )
    ap.add_argument(
        "--eval-fraction",
        type=float,
        default=0.1,
        help="Per-split stratified eval fraction (default 0.1)",
    )
    ap.add_argument(
        "--rl-train-fraction",
        type=float,
        default=0.8,
        help="Fraction of origqa data used for RL training vs holdout (default 0.8)",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Root directory; sft/ and rl/ subdirs are created here "
             "(default: input's parent directory)",
    )
    ap.add_argument(
        "--base",
        default=None,
        help="Base parquet passed to remap_fields_by_id.py (default: its own default)",
    )
    ap.add_argument("--seed", type=int, default=1, help="Random seed")
    ap.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate files under the temp dir instead of cleaning up",
    )
    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")
    if not (0 < args.refusal_fraction < 1):
        raise SystemExit("--refusal-fraction must be in (0, 1)")
    if not (0 < args.eval_fraction <= 1):
        raise SystemExit("--eval-fraction must be in (0, 1]")
    if not (0 < args.rl_train_fraction < 1):
        raise SystemExit("--rl-train-fraction must be in (0, 1)")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent
    sft_dir = out_dir / "sft"
    rl_dir = out_dir / "rl"
    sft_dir.mkdir(parents=True, exist_ok=True)
    rl_dir.mkdir(parents=True, exist_ok=True)

    base = Path(args.base).resolve() if args.base else None

    tmp = Path(tempfile.mkdtemp(prefix="refusal_gen_", dir=str(out_dir)))
    print(f"Intermediate dir: {tmp}")

    try:
        # ============================================================
        # Shared: split input and augment refusal
        # ============================================================

        # 1. Split: sampled -> refusal source, remainder -> answer portion.
        refusal_src = tmp / "refusal_source.parquet"
        answer_part = tmp / "answer.parquet"
        make_subset(
            input_path,
            refusal_src,
            args.refusal_fraction,
            args.seed,
            save_remainder=True,
            train_output=answer_part,
        )

        # 2. Rewrite the refusal-source portion with simpleqa_refusal.
        refusal_part = tmp / "refusal.parquet"
        augment_refusal(refusal_src, refusal_part, args.seed)

        # ============================================================
        # SFT outputs
        # ============================================================
        print("\n=== SFT ===")

        # 3. Combined training set = answer ∪ refusal.
        combined_tmp = combine(
            [answer_part, refusal_part],
            tmp / "sft_combined",
            shuffle=True,
            seed=args.seed,
        )
        _copy(combined_tmp, sft_dir / "train.parquet", "sft train")

        # 4. origqa: remap each side (refusal preserves targets), combine.
        answer_origqa = tmp / "answer_origqa.parquet"
        refusal_origqa = tmp / "refusal_origqa.parquet"
        remap(answer_part, answer_origqa, base, preserve_answer_reward=False)
        remap(refusal_part, refusal_origqa, base, preserve_answer_reward=True)

        combined_origqa_tmp = combine(
            [answer_origqa, refusal_origqa],
            tmp / "sft_combined_origqa",
            shuffle=True,
            seed=args.seed,
        )
        _copy(combined_origqa_tmp, sft_dir / "train_origqa.parquet", "sft train_origqa")

        # 5. Stratified eval splits (sample eval-fraction from each side).
        answer_eval = tmp / "sft_answer_eval.parquet"
        refusal_eval = tmp / "sft_refusal_eval.parquet"
        make_subset(answer_part, answer_eval, args.eval_fraction, args.seed,
                    save_remainder=False, train_output=None)
        make_subset(refusal_part, refusal_eval, args.eval_fraction, args.seed,
                    save_remainder=False, train_output=None)

        # train_eval_answer: sampled from answer_part ONLY (no refusal rows).
        _copy(answer_eval, sft_dir / "train_eval_answer.parquet", "sft eval answer")

        # train_eval_refusal: target = pre-augmentation answer from refusal_src
        # (the input's original answers for these IDs, before simpleqa_refusal).
        refusal_eval_with_target = tmp / "sft_refusal_eval_target.parquet"
        fill_target(refusal_eval, refusal_src, refusal_eval_with_target)
        _copy(refusal_eval_with_target, sft_dir / "train_eval_refusal.parquet", "sft eval refusal")

        # origqa mirrors of eval splits.
        answer_eval_origqa = tmp / "sft_answer_eval_origqa.parquet"
        refusal_eval_origqa = tmp / "sft_refusal_eval_origqa.parquet"
        remap(answer_eval, answer_eval_origqa, base, preserve_answer_reward=False)
        remap(refusal_eval, refusal_eval_origqa, base, preserve_answer_reward=True)

        _copy(answer_eval_origqa, sft_dir / "train_eval_origqa_answer.parquet",
              "sft eval origqa answer")

        # origqa refusal target = origqa version of the pre-augmentation answer.
        refusal_src_origqa = tmp / "refusal_src_origqa.parquet"
        remap(refusal_src, refusal_src_origqa, base, preserve_answer_reward=False)

        refusal_eval_origqa_target = tmp / "sft_refusal_eval_origqa_target.parquet"
        fill_target(refusal_eval_origqa, refusal_src_origqa, refusal_eval_origqa_target)
        _copy(refusal_eval_origqa_target, sft_dir / "train_eval_origqa_refusal.parquet",
              "sft eval origqa refusal")

        # ============================================================
        # RL outputs (built from the origqa splits)
        # ============================================================
        print("\n=== RL ===")

        # 6. 80/20 split of each origqa side.
        ans_rl_train = tmp / "rl_ans_train.parquet"
        ans_rl_holdout = tmp / "rl_ans_holdout.parquet"
        make_subset(answer_origqa, ans_rl_train, args.rl_train_fraction, args.seed,
                    save_remainder=True, train_output=ans_rl_holdout)

        ref_rl_train = tmp / "rl_ref_train.parquet"
        ref_rl_holdout = tmp / "rl_ref_holdout.parquet"
        make_subset(refusal_origqa, ref_rl_train, args.rl_train_fraction, args.seed,
                    save_remainder=True, train_output=ref_rl_holdout)

        # 7. Combine the 80% portions -> rl/train.parquet.
        rl_combined_tmp = combine(
            [ans_rl_train, ref_rl_train],
            tmp / "rl_combined",
            shuffle=True,
            seed=args.seed,
        )
        _copy(rl_combined_tmp, rl_dir / "train.parquet", "rl train")

        # 8. train_eval: eval-fraction sample from each 80% portion.
        rl_train_eval_ans = tmp / "rl_train_eval_ans.parquet"
        rl_train_eval_ref = tmp / "rl_train_eval_ref.parquet"
        make_subset(ans_rl_train, rl_train_eval_ans, args.eval_fraction, args.seed,
                    save_remainder=False, train_output=None)
        make_subset(ref_rl_train, rl_train_eval_ref, args.eval_fraction, args.seed,
                    save_remainder=False, train_output=None)

        _copy(rl_train_eval_ans, rl_dir / "train_eval_answer.parquet", "rl train_eval answer")

        # RL refusal target = origqa pre-augmentation answer (refusal_src_origqa).
        rl_train_eval_ref_target = tmp / "rl_train_eval_ref_target.parquet"
        fill_target(rl_train_eval_ref, refusal_src_origqa, rl_train_eval_ref_target)
        _copy(rl_train_eval_ref_target, rl_dir / "train_eval_refusal.parquet",
              "rl train_eval refusal")

        # 9. val: eval-fraction sample from each 20% holdout portion.
        rl_val_ans = tmp / "rl_val_ans.parquet"
        rl_val_ref = tmp / "rl_val_ref.parquet"
        make_subset(ans_rl_holdout, rl_val_ans, args.eval_fraction, args.seed,
                    save_remainder=False, train_output=None)
        make_subset(ref_rl_holdout, rl_val_ref, args.eval_fraction, args.seed,
                    save_remainder=False, train_output=None)

        _copy(rl_val_ans, rl_dir / "val_answer.parquet", "rl val answer")

        rl_val_ref_target = tmp / "rl_val_ref_target.parquet"
        fill_target(rl_val_ref, refusal_src_origqa, rl_val_ref_target)
        _copy(rl_val_ref_target, rl_dir / "val_refusal.parquet", "rl val refusal")

    finally:
        if not args.keep_intermediate:
            shutil.rmtree(tmp, ignore_errors=True)

    print("\nDone. Outputs:")
    print(f"  SFT:")
    print(f"    {sft_dir / 'train.parquet'}")
    print(f"    {sft_dir / 'train_origqa.parquet'}")
    print(f"    {sft_dir / 'train_eval_answer.parquet'}")
    print(f"    {sft_dir / 'train_eval_refusal.parquet'}")
    print(f"    {sft_dir / 'train_eval_origqa_answer.parquet'}")
    print(f"    {sft_dir / 'train_eval_origqa_refusal.parquet'}")
    print(f"  RL:")
    print(f"    {rl_dir / 'train.parquet'}")
    print(f"    {rl_dir / 'train_eval_answer.parquet'}")
    print(f"    {rl_dir / 'train_eval_refusal.parquet'}")
    print(f"    {rl_dir / 'val_answer.parquet'}")
    print(f"    {rl_dir / 'val_refusal.parquet'}")


if __name__ == "__main__":
    main()
