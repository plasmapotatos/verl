"""
Constrained decoding — replace bracketed factual anchors with the
highest-scoring candidate from a bag-of-facts using token-trie beam search.

Usage:
    apptainer exec --nv <container> python experiments/constrained_decoding_smoketest.py \
        --model_path <path> --bag_file <path> --input <parquet> --output <parquet>
"""

import argparse
import math
import re
import string
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize(span):
    span = span.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(sorted(span.split()))


def load_bag(path):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    return [s.strip() for s in lines if s.strip()]


def find_bracket_positions(text):
    return [(m.start(), m.end(), m.group(1))
            for m in re.finditer(r'\[([^\[\]]+)\]', text)]


_SKIP_SPAN = re.compile(
    r"^("
    r"[-+]?\d[\d,./%]*"                          # bare numbers, percentages
    r"|\d+\s*(st|nd|rd|th)\b"                     # ordinals
    r"|[$€£¥₹][\d,.]+.*"                          # money
    r"|v?\d+\.\d+[\w.-]*"                         # version strings
    r"|\d{4}s?"                                   # years
    r"|\d{1,2}\s+\w+\s+\d{4}"                    # "28 March 1997"
    r"|\w+\s+\d{1,2},?\s+\d{4}"                  # "March 28, 1997"
    r"|\w+\s+\d{4}"                               # "May 2021"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"                  # "3/15/2020"
    r"|\d{4}\s*(to|[-–—])\s*\d{4}"               # "1837 to 1839", "1837-1839"
    r"|\d+\s*(year|month|day|week|hour|minute|second|km|miles?|feet|meters?|kg|pounds?|tons?|acres?|hectares?)s?"
    r")$",
    re.IGNORECASE,
)

_SKIP_EXACT = {
    'a', 'an', 'the', 'his', 'her', 'its', 'their', 'they', 'them',
    'he', 'she', 'it', 'we', 'us', 'our', 'this', 'that', 'these', 'those',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
    'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'hundred',
    'thousand', 'million', 'billion',
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
    'eighth', 'ninth', 'tenth', 'last',
}


def is_non_entity_span(span):
    """Return True for spans that should NOT be replaced (dates, numbers, etc.)."""
    s = span.strip()
    if not s:
        return True
    if s.lower() in _SKIP_EXACT:
        return True
    if _SKIP_SPAN.match(s):
        return True
    if len(s) <= 2 and not s.isalpha():
        return True
    return False


def compute_idf(parquet_path, answer_col="answer"):
    """Compute IDF scores from a parquet where each row is one fact/question.

    Returns dict mapping normalized anchor -> IDF score (log(N/df)).
    """
    df = pd.read_parquet(parquet_path)
    if answer_col not in df.columns:
        raise KeyError(f"Expected column '{answer_col}' in {parquet_path}")

    doc_freq = {}
    n_docs = len(df)
    for answer in df[answer_col].tolist():
        spans = set()
        for m in re.findall(r'\[([^\[\]]+)\]', str(answer or '')):
            spans.add(normalize(m.strip()))
        for s in spans:
            doc_freq[s] = doc_freq.get(s, 0) + 1

    idf = {}
    for span, df_count in doc_freq.items():
        idf[span] = math.log(n_docs / df_count)
    return idf


# ── trie + beam search ──────────────────────────────────────────────────────

def build_anchor_trie(tokenizer, spans):
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
def decode_from_trie_with_beam(
    prefix_text, trie_root, model, tokenizer,
    beam_size, max_steps, length_norm_alpha,
):
    device = next(model.parameters()).device
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    beams = [{'node': trie_root, 'tokens': [], 'score': 0.0}]
    finished = []

    for _ in range(max_steps):
        active = [b for b in beams if b['node']['children']]
        for b in beams:
            if b['node']['terminal_texts'] and b['tokens']:
                finished.append(b)
        if not active:
            break

        seqs = [prefix_ids + b['tokens'] for b in active]
        max_len = max(len(s) for s in seqs)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        for i, seq in enumerate(seqs):
            n = len(seq)
            input_ids[i, -n:] = torch.tensor(seq, dtype=torch.long, device=device)
            attn[i, -n:] = 1

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        last_positions = torch.tensor([max_len - 1] * len(seqs), device=device)
        next_logprobs = torch.log_softmax(
            logits[torch.arange(len(seqs), device=device), last_positions].float(), dim=-1
        )

        expansions = []
        for i, b in enumerate(active):
            for tok_id, child in b['node']['children'].items():
                expansions.append({
                    'node': child,
                    'tokens': b['tokens'] + [tok_id],
                    'score': b['score'] + next_logprobs[i, tok_id].item(),
                })
        if not expansions:
            break

        expansions.sort(key=lambda x: x['score'], reverse=True)
        beams = expansions[:beam_size]

    for b in beams:
        if b['node']['terminal_texts'] and b['tokens']:
            finished.append(b)

    if not finished:
        return []

    best_by_text = {}
    for cand in finished:
        text = cand['node']['terminal_texts'][0]
        prev = best_by_text.get(text)
        if prev is None or cand['score'] > prev['score']:
            best_by_text[text] = cand

    results = []
    for text, cand in best_by_text.items():
        n_tok = len(cand['tokens'])
        rank_denom = max(float(n_tok) ** float(length_norm_alpha), 1.0)
        results.append((text, cand['score'], cand['score'] / max(n_tok, 1), n_tok,
                         cand['score'] / rank_denom))

    results.sort(key=lambda x: x[4], reverse=True)
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--bag_file", required=True, help="Bag-of-facts txt file, one span per line.")
    parser.add_argument("--input", required=True, help="Input generations parquet.")
    parser.add_argument("--output", default=None,
                        help="Output parquet (default: <input_dir>/constrained_decoding.parquet).")
    parser.add_argument("--beam_size", type=int, default=8)
    parser.add_argument("--length_norm_alpha", type=float, default=0.7)
    parser.add_argument("--idf_corpus", default=None,
                        help="Source parquet for IDF computation (needs 'answer' column with [brackets]).")
    parser.add_argument("--idf_threshold", type=float, default=0.0,
                        help="Drop bag anchors with IDF below this value (0 = no filtering).")
    parser.add_argument("--idf_weight", type=float, default=0.0,
                        help="Blend IDF into ranking: final_score = rank_score + idf_weight * idf. (0 = no reweighting).")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation after generation.")
    parser.add_argument("--eval_dataset", default="simpleqa", help="Dataset for eval (default: simpleqa).")
    parser.add_argument("--eval_workers", type=int, default=32, help="Eval worker count (default: 32).")
    parser.add_argument("--judge_model", default="gpt-4o-mini", help="Judge model (default: gpt-4o-mini).")
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(args.input).parent / "constrained_decoding.parquet")

    # ── load data ────────────────────────────────────────────────────────────
    bag_spans = load_bag(args.bag_file)
    bag_norm = set(normalize(s) for s in bag_spans)
    print(f"Bag: {len(bag_spans)} spans ({len(bag_norm)} unique normalized)")

    idf_scores = {}
    if args.idf_corpus:
        idf_scores = compute_idf(args.idf_corpus)
        print(f"IDF: computed for {len(idf_scores)} unique anchors from {args.idf_corpus}")
        idf_vals = sorted(idf_scores.values())
        print(f"  range: [{idf_vals[0]:.2f}, {idf_vals[-1]:.2f}], "
              f"median: {idf_vals[len(idf_vals)//2]:.2f}")

        if args.idf_threshold > 0:
            before = len(bag_spans)
            bag_spans = [s for s in bag_spans
                         if idf_scores.get(normalize(s), float('inf')) >= args.idf_threshold]
            bag_norm = set(normalize(s) for s in bag_spans)
            print(f"  IDF filter (>= {args.idf_threshold}): {before} -> {len(bag_spans)} spans")

    df = pd.read_parquet(args.input)
    print(f"Input: {len(df)} rows from {args.input}")

    # ── load model + build trie ──────────────────────────────────────────────
    print(f"Loading model from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    trie_root, trie_max_depth = build_anchor_trie(tokenizer, bag_spans)
    print(f"Trie ready (max depth={trie_max_depth})")

    # ── process rows ─────────────────────────────────────────────────────────
    total_spans = 0
    bag_hits = 0
    skipped_non_entity = 0
    replacements = 0
    out_responses = []

    for row_idx, row in df.iterrows():
        responses = row['responses']
        response = responses[0] if hasattr(responses, '__getitem__') and len(responses) > 0 else ''

        brackets = find_bracket_positions(response)
        if not brackets:
            out_responses.append(responses)
            continue

        prompt = row.get('prompt', row.get('question', ''))
        if isinstance(prompt, list):
            prompt_text = tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False,
            )
        else:
            prompt_text = str(prompt)

        new_response = response
        offset = 0

        for start, end, span_text in brackets:
            total_spans += 1

            if normalize(span_text) in bag_norm:
                bag_hits += 1
                continue

            if is_non_entity_span(span_text):
                skipped_non_entity += 1
                continue

            adj_start = start + offset
            prefix = prompt_text + new_response[:adj_start + 1]

            scored = decode_from_trie_with_beam(
                prefix, trie_root, model, tokenizer,
                beam_size=args.beam_size,
                max_steps=trie_max_depth,
                length_norm_alpha=args.length_norm_alpha,
            )
            if not scored:
                continue

            if args.idf_weight > 0 and idf_scores:
                scored = [
                    (text, total_lp, avg_lp, n_tok,
                     rank_score + args.idf_weight * idf_scores.get(normalize(text), 0.0))
                    for text, total_lp, avg_lp, n_tok, rank_score in scored
                ]
                scored.sort(key=lambda x: x[4], reverse=True)

            best_span = scored[0][0]
            old_bracket = f"[{span_text}]"
            new_bracket = f"[{best_span}]"
            new_response = (new_response[:adj_start]
                            + new_bracket
                            + new_response[adj_start + len(old_bracket):])
            offset += len(new_bracket) - len(old_bracket)
            replacements += 1

            print(f"  [{row_idx:3d}] [{span_text}] -> [{best_span}] "
                  f"(score={scored[0][4]:.2f})")

        out_responses.append([new_response])

        if (row_idx + 1) % 50 == 0:
            print(f"  ... {row_idx + 1}/{len(df)} rows")

    # ── save ─────────────────────────────────────────────────────────────────
    df = df.copy()
    df['responses'] = out_responses
    df.to_parquet(args.output, index=False)

    print()
    print(f"Total bracket spans: {total_spans}")
    print(f"  Bag hits (kept):   {bag_hits}")
    print(f"  Skipped (non-entity): {skipped_non_entity}")
    print(f"  Replacements:      {replacements}")
    print(f"Saved to {args.output}")

    if args.skip_eval:
        return

    # ── eval ─────────────────────────────────────────────────────────────────
    output_stem = args.output[:-8] if args.output.endswith('.parquet') else args.output

    rule_output = output_stem + ".eval.rule.json"
    print(f"\nRunning rule eval -> {rule_output}")
    subprocess.run([
        sys.executable, "-m", "verl.eval.cli",
        "--dataset", args.eval_dataset,
        "--input", args.output,
        "--output", rule_output,
        "--workers", str(args.eval_workers),
    ], check=True)

    judge_output = output_stem + ".eval.judge.json"
    print(f"Running judge eval ({args.judge_model}) -> {judge_output}")
    subprocess.run([
        sys.executable, "-m", "verl.eval.cli",
        "--dataset", args.eval_dataset,
        "--input", args.output,
        "--output", judge_output,
        "--workers", str(args.eval_workers),
        "--use-judge",
        "--judge-model", args.judge_model,
    ], check=True)

    print("Eval complete.")


if __name__ == "__main__":
    main()
