#!/usr/bin/env python3
"""Step 1: Build a bracket-substituted SFT dataset.

For each RL training question:
    1. Generate a greedy rollout from the model.
    2. Extract [...] spans from the rollout.
    3. For each span, decode a bag-constrained replacement using token-trie
        beam search conditioned on the prefix ending at ``[``.
    4. Substitute the span with the top-scoring fact (fall back to the
         rollout's original span if no bag fact beats it).
    5. Record the token positions of all bracket content spans in the
         modified rollout so Step 2 can build a loss mask.

Output is a JSONL in ``<generations_dir>/bracket_sft_data/``.

Example:

        apptainer exec --nv /work/hdd/bbsg/twei2/rl/torch2501.sif \\
                python scripts/build_bracket_sft_data.py \\
                --model_path /work/.../global_step_570/merged_hf_model \\
                --bag_file   /work/.../generations/..._on_train_eval_eval.json \\
                --input_questions /work/.../smoketest/rl/train_unbracketed.parquet \\
                --limit 32
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


# ── bracket helpers ──────────────────────────────────────────────────────────

def extract_brackets(text: str) -> List[str]:
    return [m.group(1) for m in BRACKET_RE.finditer(text or "")]


def find_bracket_char_spans(text: str):
    """Char positions of each ``[...]`` in *text*.

    Returns a list of dicts with keys: ``open`` (index of ``[``),
    ``content_start`` (char just after ``[``), ``content_end`` (index of
    ``]``), ``close`` (one past ``]``), and ``text`` (content).
    """
    out = []
    for m in BRACKET_RE.finditer(text or ""):
        out.append(
            {
                "open": m.start(),
                "content_start": m.start() + 1,
                "content_end": m.end() - 1,
                "close": m.end(),
                "text": m.group(1),
            }
        )
    return out


# ── bag loading ──────────────────────────────────────────────────────────────

def _bag_from_json_rows(rows, bag: Counter) -> None:
    for row in rows:
        for span in extract_brackets(row.get("answer", "")):
            s = span.strip()
            if s:
                bag[s] += 1


def load_bag(path: Path) -> List[str]:
    """Load bag from a ``.txt`` file, a generations JSON, or a generations dir.

    - ``.txt``: one fact per line (matches ``extract_factual_anchors.py`` output).
    - ``.json``: an eval/generations file — extract ``[...]`` from each row's
      ``answer`` field.
    - directory: merge all ``*_on_train_eval_eval.json`` files (falls back to
      ``*_eval.json``).

    Entries are deduped by exact string. For ``.json`` / directory inputs
    the list is sorted by frequency (most common first); for ``.txt`` the
    file order is preserved.
    """
    if path.is_file() and path.suffix.lower() == ".txt":
        seen = set()
        out: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    bag: Counter = Counter()
    if path.is_file():
        with open(path) as f:
            data = json.load(f)
        _bag_from_json_rows(data.get("rows", []), bag)
    else:
        files = sorted(path.glob("*_on_train_eval_eval.json"))
        if not files:
            files = sorted(path.glob("*_eval.json"))
        if not files:
            raise FileNotFoundError(f"No eval JSON files under {path}")
        for f in files:
            with open(f) as fh:
                data = json.load(fh)
            _bag_from_json_rows(data.get("rows", []), bag)
    return [s for s, _ in bag.most_common()]


# ── question loading ─────────────────────────────────────────────────────────

def load_questions(path: Path, limit: int | None = None):
    df = pd.read_parquet(path)
    if limit:
        df = df.head(limit)
    rows = []
    for _, r in df.iterrows():
        prompt = r.get("prompt")
        if (
            prompt is not None
            and hasattr(prompt, "__len__")
            and len(prompt) > 0
            and isinstance(prompt[0], dict)
        ):
            messages = [dict(m) for m in prompt]
        else:
            messages = [{"role": "user", "content": str(r.get("question", ""))}]
        rows.append(
            {
                "id": str(r.get("id", "")),
                "question": messages[-1].get("content", ""),
                "messages": messages,
            }
        )
    return rows


# ── generation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def greedy_generate_batch(
    model,
    tokenizer,
    prompt_texts: List[str],
    max_new_tokens: int,
    device,
) -> List[str]:
    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)
    out = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_len = enc["input_ids"].shape[1]
    gen_ids = out[:, input_len:]
    responses = []
    for row in gen_ids:
        tokens = row.tolist()
        if tokenizer.eos_token_id in tokens:
            tokens = tokens[: tokens.index(tokenizer.eos_token_id)]
        responses.append(tokenizer.decode(tokens, skip_special_tokens=True))
    return responses


# ── scoring ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_candidates_batched(
    model,
    tokenizer,
    prefix_ids: List[int],
    cand_token_lists: List[List[int]],
    batch_size: int,
    device,
) -> List[float]:
    """For each candidate c, return sum_i log P(c[i] | prefix, c[:i]).

    One forward pass per mini-batch. Uses left-padding so the final
    ``len(prefix)+len(c)`` tokens of each row are meaningful.
    """
    if not cand_token_lists:
        return []
    pad_id = tokenizer.pad_token_id
    prefix_len = len(prefix_ids)
    results: List[float] = []

    for i in range(0, len(cand_token_lists), batch_size):
        chunk = cand_token_lists[i : i + batch_size]
        total_lens = [prefix_len + len(c) for c in chunk]
        max_total = max(total_lens)
        B = len(chunk)

        input_ids = torch.full((B, max_total), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((B, max_total), dtype=torch.long, device=device)
        for j, c in enumerate(chunk):
            full = prefix_ids + c
            n = len(full)
            input_ids[j, -n:] = torch.as_tensor(full, dtype=torch.long, device=device)
            attn[j, -n:] = 1

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        for j, c in enumerate(chunk):
            if not c:
                results.append(0.0)
                continue
            n = prefix_len + len(c)
            offset = max_total - n
            # Position (offset + prefix_len - 1) predicts c[0];
            # position (offset + prefix_len - 1 + k) predicts c[k].
            start = offset + prefix_len - 1
            positions = torch.arange(start, start + len(c), device=device)
            tok = torch.as_tensor(c, dtype=torch.long, device=device)
            lp = logprobs[j].index_select(0, positions).gather(1, tok.unsqueeze(1)).sum().item()
            results.append(lp)

    return results


def build_anchor_trie(tokenizer, spans: List[str]):
    """Build a token trie from candidate spans.

    Returns ``(root, max_depth)`` where each node contains:
      - ``children``: token_id -> node
      - ``terminal_texts``: list[str]
    """
    root = {"children": {}, "terminal_texts": []}
    max_depth = 0

    for text in spans:
        tok = tokenizer.encode(text, add_special_tokens=False)
        if not tok:
            continue
        node = root
        for t in tok:
            node = node["children"].setdefault(t, {"children": {}, "terminal_texts": []})
        node["terminal_texts"].append(text)
        max_depth = max(max_depth, len(tok))

    return root, max_depth


@torch.no_grad()
def decode_best_from_trie(
    model,
    tokenizer,
    prefix_ids: List[int],
    trie_root,
    max_steps: int,
    beam_size: int,
    device,
):
    """Decode best bag candidate under trie constraints.

    Returns ``(text, total_logprob)`` or ``None`` if no terminal is reachable.
    """
    beams = [{"node": trie_root, "tokens": [], "score": 0.0}]
    finished = []
    beam_size = max(1, beam_size)

    for _ in range(max_steps):
        active = [b for b in beams if b["node"]["children"]]

        for b in beams:
            if b["node"]["terminal_texts"] and b["tokens"]:
                finished.append(b)

        if not active:
            break

        seqs = [prefix_ids + b["tokens"] for b in active]
        max_len = max(len(s) for s in seqs)
        pad_id = tokenizer.pad_token_id

        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(seqs):
            n = len(seq)
            input_ids[i, -n:] = torch.as_tensor(seq, dtype=torch.long, device=device)
            attn[i, -n:] = 1

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        next_logprobs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)

        expansions = []
        for i, b in enumerate(active):
            for tok_id, child in b["node"]["children"].items():
                lp = next_logprobs[i, tok_id].item()
                expansions.append(
                    {
                        "node": child,
                        "tokens": b["tokens"] + [tok_id],
                        "score": b["score"] + lp,
                    }
                )

        if not expansions:
            break

        expansions.sort(key=lambda x: x["score"], reverse=True)
        beams = expansions[:beam_size]

    for b in beams:
        if b["node"]["terminal_texts"] and b["tokens"]:
            finished.append(b)

    if not finished:
        return None

    best = max(finished, key=lambda x: x["score"])
    best_text = best["node"]["terminal_texts"][0]
    return best_text, best["score"]


# ── per-question substitution ────────────────────────────────────────────────

def process_response(
    model,
    tokenizer,
    prompt_text: str,
    response_text: str,
    trie_root,
    trie_max_depth: int,
    beam_size: int,
    batch_size: int,
    device,
):
    spans = find_bracket_char_spans(response_text)
    if not spans:
        return response_text, [], 0, 0, 0, []

    new_response = response_text
    offset = 0
    substitutions = []
    n_sub = 0
    n_fallback = 0
    gaps: List[float] = []

    for sp in spans:
        orig_span = sp["text"]
        open_char = sp["open"] + offset
        content_start = sp["content_start"] + offset
        content_end = sp["content_end"] + offset

        prefix_text = prompt_text + new_response[: open_char + 1]
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

        best = decode_best_from_trie(
            model,
            tokenizer,
            prefix_ids,
            trie_root,
            trie_max_depth,
            beam_size,
            device,
        )
        if best is None:
            n_fallback += 1
            substitutions.append(
                {
                    "original": orig_span,
                    "replacement": orig_span,
                    "score": None,
                    "original_score": None,
                    "gap": None,
                    "fallback": True,
                }
            )
            continue

        best_text, best_score = best

        orig_ids = tokenizer.encode(orig_span, add_special_tokens=False)
        orig_score = score_candidates_batched(
            model, tokenizer, prefix_ids, [orig_ids], batch_size, device
        )[0]
        gap = best_score - orig_score
        gaps.append(gap)

        if best_score > orig_score and best_text != orig_span:
            new_response = (
                new_response[:content_start] + best_text + new_response[content_end:]
            )
            offset += len(best_text) - len(orig_span)
            substitutions.append(
                {
                    "original": orig_span,
                    "replacement": best_text,
                    "score": round(best_score, 4),
                    "original_score": round(orig_score, 4),
                    "gap": round(gap, 4),
                    "fallback": False,
                }
            )
            n_sub += 1
        else:
            n_fallback += 1
            substitutions.append(
                {
                    "original": orig_span,
                    "replacement": orig_span,
                    "score": round(best_score, 4),
                    "original_score": round(orig_score, 4),
                    "gap": round(gap, 4),
                    "fallback": True,
                }
            )

    return new_response, substitutions, len(spans), n_sub, n_fallback, gaps


def compute_bracket_token_spans(tokenizer, response_text: str) -> List[Tuple[int, int]]:
    """Half-open (start, end) token-index pairs for each ``[...]`` content.

    Indices are relative to a fresh tokenization of ``response_text`` with
    ``add_special_tokens=False``. Only tokens whose char range lies strictly
    inside the brackets are counted.
    """
    enc = tokenizer(
        response_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    token_spans: List[Tuple[int, int]] = []
    for sp in find_bracket_char_spans(response_text):
        cs, ce = sp["content_start"], sp["content_end"]
        start_tok, end_tok = None, None
        for idx, (a, b) in enumerate(offsets):
            if a == b:
                continue
            if a >= cs and b <= ce:
                if start_tok is None:
                    start_tok = idx
                end_tok = idx + 1
        if start_tok is not None:
            token_spans.append((start_tok, end_tok))
    return token_spans


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--bag_file",
        required=True,
        help=(
            "Bag of facts source. Accepts: (a) a .txt file with one fact per line "
            "(e.g. output of scripts/extract_factual_anchors.py); "
            "(b) a generations eval JSON — [...] spans are extracted from each "
            "row's `answer`; (c) a directory of eval JSONs."
        ),
    )
    parser.add_argument(
        "--input_questions",
        required=True,
        help="Parquet with RL train split questions (unbracketed, same format as RL training).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Defaults to <bag_file dir>/bracket_sft_data/.",
    )
    parser.add_argument("--output_name", default="bracket_sft_data.parquet")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gen_batch_size", type=int, default=8)
    parser.add_argument("--score_batch_size", type=int, default=64)
    parser.add_argument(
        "--beam_size",
        type=int,
        default=8,
        help="Beam width for trie-constrained span decoding.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only process the first N questions (for smoke tests).",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag_file)
    gens_dir = bag_path if bag_path.is_dir() else bag_path.parent
    out_dir = Path(args.output_dir) if args.output_dir else (gens_dir / "bracket_sft_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name
    stats_path = out_dir / (Path(args.output_name).stem + ".stats.json")

    print(f"Loading bag from {bag_path} ...")
    bag_texts = load_bag(bag_path)
    print(f"  bag size: {len(bag_texts)} unique span texts")

    print(f"Loading questions from {args.input_questions} ...")
    questions = load_questions(
        Path(args.input_questions),
        limit=args.limit if args.limit > 0 else None,
    )
    print(f"  questions: {len(questions)}")

    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  device: {device}")

    print("Building token trie for bag candidates ...")
    trie_root, trie_max_depth = build_anchor_trie(tokenizer, bag_texts)
    print(f"  trie max token depth: {trie_max_depth}")

    # Phase 1: batched greedy rollouts.
    print("Generating rollouts ...")
    prompt_texts = [
        tokenizer.apply_chat_template(
            q["messages"], add_generation_prompt=True, tokenize=False
        )
        for q in questions
    ]
    rollouts: List[str] = []
    B = max(1, args.gen_batch_size)
    for i in range(0, len(prompt_texts), B):
        batch = prompt_texts[i : i + B]
        rollouts.extend(
            greedy_generate_batch(model, tokenizer, batch, args.max_new_tokens, device)
        )
        done = min(i + B, len(prompt_texts))
        print(f"  generated {done}/{len(prompt_texts)}")

    # Phase 2: per-span scoring + substitution.
    print("Scoring + substituting spans ...")
    total_spans = 0
    total_sub = 0
    total_fallback = 0
    responses_with_spans = 0
    all_gaps: List[float] = []
    records: List[dict] = []

    for idx, (q, prompt_text, rollout) in enumerate(
        tqdm(
            zip(questions, prompt_texts, rollouts),
            total=len(questions),
            desc="Scoring/Substituting",
        )
    ):
        new_resp, subs, nsp, ns, nf, gaps = process_response(
            model,
            tokenizer,
            prompt_text,
            rollout,
            trie_root,
            trie_max_depth,
            args.beam_size,
            args.score_batch_size,
            device,
        )
        total_spans += nsp
        total_sub += ns
        total_fallback += nf
        if nsp > 0:
            responses_with_spans += 1
        all_gaps.extend(gaps)

        token_spans = compute_bracket_token_spans(tokenizer, new_resp)

        records.append(
            {
                "id": q["id"],
                # `prompt` / `response` are the columns SFTDataset reads.
                "prompt": q["question"],
                "response": new_resp,
                # list<list<int>> — parquet-friendly, length-2 inner lists.
                "bracket_token_spans": [list(s) for s in token_spans],
                "original_response": rollout,
                # nested dicts round-trip cleanly if serialized as JSON text.
                "substitutions": json.dumps(subs, ensure_ascii=False),
            }
        )

        if (idx + 1) % 25 == 0 or (idx + 1) == len(questions):
            print(
                f"  [{idx+1}/{len(questions)}] spans={total_spans} "
                f"sub={total_sub} fallback={total_fallback}"
            )

    pd.DataFrame(records).to_parquet(out_path, index=False)

    mean_gap = (sum(all_gaps) / len(all_gaps)) if all_gaps else 0.0
    stats = {
        "n_questions": len(questions),
        "responses_with_any_span": responses_with_spans,
        "total_spans": total_spans,
        "substituted": total_sub,
        "fallback_to_original": total_fallback,
        "mean_score_gap_winner_minus_original": round(mean_gap, 4),
        "bag_size": len(bag_texts),
        "model_path": args.model_path,
        "bag_file": str(bag_path),
        "input_questions": str(args.input_questions),
        "max_new_tokens": args.max_new_tokens,
    }
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(json.dumps(stats, indent=2))
    print(f"\nWrote: {out_path}")
    print(f"Wrote: {stats_path}")


if __name__ == "__main__":
    main()
