#!/usr/bin/env python3
"""
Bracketing ablation experiment.

Compares two SFT checkpoints (bracketed vs unbracketed) on the same
training-eval QA questions.  For each model we:

  1. Build a "bag of facts" from the SFT *target* answers.
       - Bracketed model  : extract [..] spans directly from targets.
       - Unbracketed model: apply GPT-4o-mini annotation to plain-text targets.

  2. Extract candidate factual spans from model *responses* on origQA.
       - Bracketed model, primary   : extract [..] spans.
       - Bracketed model, NER ctrl  : spacy en_core_web_sm entities.
       - Unbracketed model, primary : apply GPT-4o-mini annotation.

  3. For each (model, extractor) configuration, compute the fraction of
     response spans that are in the bag of facts, stratified by grader
     outcome (correct / incorrect / correct-fact-unbracketed).

Results are printed to stdout and written to:
  experiments/bracket_ablation_results.json

GPT-4o-mini calls are cached in:
  experiments/bracket_ablation_gpt_cache.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import string
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]

BRACK_DIR  = ROOT / "outputs/sft/simpleqa_factual_anchor_sft_smoketest/sft_lr1.5e-4_epmax30_seed1/global_step_570/generations"
UNBRACK_DIR = ROOT / "outputs/sft/simpleqa_factual_anchor_sft_smoketest_original/sft_lr1.5e-4_epmax30_seed1/global_step_570/generations"

RESULTS_PATH = ROOT / "experiments/bracket_ablation_results.json"
CACHE_PATH   = ROOT / "experiments/bracket_ablation_gpt_cache.json"

PROMPT_PATH = ROOT / "verl/augment/prompts/factual_anchor_brackets.txt"


# ── text utilities ─────────────────────────────────────────────────────────────

def extract_brackets(text: str) -> List[str]:
    return re.findall(r"\[([^\[\]]+)\]", text)


def normalize(span: str) -> str:
    """Lowercase, strip punctuation, sort tokens."""
    span = span.lower()
    span = span.translate(str.maketrans("", "", string.punctuation))
    return " ".join(sorted(span.split()))


def normalize_simple(text: str) -> str:
    """Lowercase + strip punctuation, preserve order (for substring search)."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def in_bag_exact(span: str, bag_set: set) -> bool:
    return normalize(span) in bag_set


def in_bag_fuzzy(span: str, bag_list: List[str], threshold: int = 85) -> bool:
    try:
        from rapidfuzz import fuzz
        n = normalize(span)
        return any(fuzz.ratio(n, b) >= threshold for b in bag_list)
    except ImportError:
        return False


# ── GPT-4o-mini annotation ────────────────────────────────────────────────────

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class AnnotationCache:
    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, str] = {}
        if path.exists():
            with open(path) as f:
                self._data = json.load(f)
            log.info(f"Loaded {len(self._data)} cached GPT annotations from {path}")

    def get(self, text: str) -> Optional[str]:
        return self._data.get(_text_hash(text))

    def set(self, text: str, result: str) -> None:
        self._data[_text_hash(text)] = result

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f)


def _build_openai_client(model: str = "gpt-4o-mini"):
    """Build OpenAIClient using the codebase's wrapper."""
    sys.path.insert(0, str(ROOT))
    from verl.augment.openai_client import OpenAIClient
    return OpenAIClient(model, timeout=90.0, retry_attempts=3, retry_delay=2.0)


def annotate_texts_gpt(
    texts: List[str],
    cache: AnnotationCache,
    system_prompt: str,
    *,
    max_workers: int = 8,
) -> List[str]:
    """
    Annotate each text with GPT-4o-mini.  Results are cached; only uncached
    texts trigger API calls.  Returns a list aligned with `texts`.
    """
    results = [None] * len(texts)
    uncached_indices = []
    for i, t in enumerate(texts):
        cached = cache.get(t)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        log.info("All texts already cached — no API calls needed.")
        return results  # type: ignore[return-value]

    log.info(f"Calling GPT-4o-mini for {len(uncached_indices)} uncached texts "
             f"(already cached: {len(texts) - len(uncached_indices)})")

    client = _build_openai_client()

    def annotate_one(idx: int) -> Tuple[int, str]:
        text = texts[idx]
        raw = client.generate(
            system_prompt=system_prompt,
            user_prompt=f"Text: {text}\nOutput:",
            seed=42,
        )
        return idx, raw.strip() or f"[{text}]"

    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(annotate_one, i): i for i in uncached_indices}
        done = 0
        for fut in as_completed(futures):
            done += 1
            if done % 20 == 0 or done == len(uncached_indices):
                log.info(f"  {done}/{len(uncached_indices)} done")
            try:
                idx, result = fut.result()
                results[idx] = result
                cache.set(texts[idx], result)
            except Exception as exc:
                idx = futures[fut]
                log.warning(f"GPT annotation failed for idx={idx}: {exc}")
                results[idx] = f"[{texts[idx][:80]}]"
                errors += 1

    cache.save()
    if errors:
        log.warning(f"{errors} GPT annotation errors (fallback applied)")
    return results  # type: ignore[return-value]


# ── spacy NER extraction ───────────────────────────────────────────────────────

def load_spacy():
    import spacy
    return spacy.load("en_core_web_sm")


def extract_ner_spans(text: str, nlp) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents]


# ── data loading ───────────────────────────────────────────────────────────────

def load_eval_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def get_category(row: dict) -> str:
    """Return grader outcome for an origQA row."""
    grader = (row.get("graders") or [{}])[0]
    return grader.get("evaluation", "unknown")


def correct_fact_unbracketed(row: dict) -> bool:
    """True if the gold fact appears in the response but NOT inside brackets."""
    gold_spans = extract_brackets(row["answer"])
    if not gold_spans:
        return False
    response = (row.get("responses") or [""])[0]
    resp_no_brackets = re.sub(r"\[[^\[\]]+\]", " ", response)
    for gs in gold_spans:
        if normalize_simple(gs) and normalize_simple(gs) in normalize_simple(resp_no_brackets):
            return True
    return False


def classify_row(row: dict) -> str:
    base = get_category(row)
    if base == "incorrect" and correct_fact_unbracketed(row):
        return "correct_fact_unbracketed"
    return base


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    span_lists: List[List[str]],
    categories: List[str],
    bag_set: set,
    bag_list: List[str],
) -> dict:
    """
    span_lists[i] = list of candidate spans for origQA row i.
    categories[i] = 'correct' | 'incorrect' | 'correct_fact_unbracketed' | ...

    Returns nested dict: category -> span-level and response-level counts.
    """
    from collections import defaultdict

    span_stats = defaultdict(lambda: {"total": 0, "exact": 0, "fuzzy": 0, "none": 0})
    resp_stats = defaultdict(lambda: {"n": 0, "has_span": 0, "all_exact": 0, "all_inbag": 0})

    for spans, cat in zip(span_lists, categories):
        resp_stats[cat]["n"] += 1
        if not spans:
            continue
        resp_stats[cat]["has_span"] += 1

        row_exact = True
        row_inbag = True
        for sp in spans:
            exact = in_bag_exact(sp, bag_set)
            fuzzy = (not exact) and in_bag_fuzzy(sp, bag_list)
            span_stats[cat]["total"] += 1
            if exact:
                span_stats[cat]["exact"] += 1
            elif fuzzy:
                span_stats[cat]["fuzzy"] += 1
            else:
                span_stats[cat]["none"] += 1
                row_inbag = False
            if not exact:
                row_exact = False

        if row_exact:
            resp_stats[cat]["all_exact"] += 1
        if row_inbag:
            resp_stats[cat]["all_inbag"] += 1

    # Aggregate totals
    all_cats = list(span_stats.keys())
    total_spans = sum(span_stats[c]["total"] for c in all_cats)
    total_exact = sum(span_stats[c]["exact"] for c in all_cats)
    total_fuzzy = sum(span_stats[c]["fuzzy"] for c in all_cats)

    return {
        "span_stats": dict(span_stats),
        "resp_stats": dict(resp_stats),
        "overall": {
            "total_spans": total_spans,
            "exact": total_exact,
            "fuzzy": total_fuzzy,
            "inbag": total_exact + total_fuzzy,
            "inbag_pct": (total_exact + total_fuzzy) / total_spans * 100 if total_spans else 0,
        },
    }


# ── printing ──────────────────────────────────────────────────────────────────

CAT_ORDER = ["correct", "incorrect", "correct_fact_unbracketed", "not_attempted", "unknown"]


def print_metrics(label: str, metrics: dict, bag_size: int) -> None:
    ov = metrics["overall"]
    ss = metrics["span_stats"]
    rs = metrics["resp_stats"]

    print(f"\n{'=' * 76}")
    print(f"  {label}  (bag size: {bag_size} unique normalized spans)")
    print(f"{'=' * 76}")
    print(f"  Total response spans : {ov['total_spans']}")
    print(f"  Exact in-bag         : {ov['exact']} ({ov['exact']/ov['total_spans']*100:.1f}%)" if ov["total_spans"] else "  (no spans)")
    print(f"  Fuzzy in-bag (≥85)   : {ov['fuzzy']} ({ov['fuzzy']/ov['total_spans']*100:.1f}%)" if ov["total_spans"] else "")
    print(f"  Combined in-bag      : {ov['inbag']} ({ov['inbag_pct']:.1f}%)")

    print(f"\n  Span level by category:")
    print(f"  {'Category':<28} {'Spans':>6} {'Exact%':>8} {'Fuzzy%':>8} {'InBag%':>8} {'None%':>7}")
    print(f"  {'-' * 70}")
    for cat in CAT_ORDER:
        if cat not in ss:
            continue
        s = ss[cat]
        t = s["total"]
        if t == 0:
            continue
        e_p = s["exact"] / t * 100
        f_p = s["fuzzy"] / t * 100
        n_p = s["none"] / t * 100
        print(f"  {cat:<28} {t:>6} {e_p:>7.1f}% {f_p:>7.1f}% {e_p+f_p:>7.1f}% {n_p:>6.1f}%")

    print(f"\n  Response level by category:")
    print(f"  {'Category':<28} {'Resp':>6} {'HasSpan':>9} {'AllExact':>10} {'AllInBag':>10}")
    print(f"  {'-' * 70}")
    for cat in CAT_ORDER:
        if cat not in rs:
            continue
        r = rs[cat]
        n = r["n"]
        if n == 0:
            continue
        print(f"  {cat:<28} {n:>6} {r['has_span']/n*100:>8.1f}% {r['all_exact']/n*100:>9.1f}% {r['all_inbag']/n*100:>9.1f}%")


def print_comparison_table(results: dict) -> None:
    configs = list(results.keys())
    print(f"\n{'=' * 76}")
    print("  SUMMARY COMPARISON — Overall in-bag % (span level)")
    print(f"{'=' * 76}")
    print(f"  {'Configuration':<45} {'Spans':>6} {'InBag%':>8}")
    print(f"  {'-' * 65}")
    for cfg, res in results.items():
        ov = res["metrics"]["overall"]
        print(f"  {cfg:<45} {ov['total_spans']:>6} {ov['inbag_pct']:>7.1f}%")

    print(f"\n  SUMMARY COMPARISON — In-bag % by correctness (span level)")
    print(f"  {'Configuration':<45} {'correct':>10} {'incorrect':>11} {'c_unbkt':>9}")
    print(f"  {'-' * 80}")
    for cfg, res in results.items():
        ss = res["metrics"]["span_stats"]

        def pct(cat):
            s = ss.get(cat, {})
            t = s.get("total", 0)
            if t == 0:
                return "  n/a"
            return f"{(s['exact']+s['fuzzy'])/t*100:>7.1f}%"

        print(f"  {cfg:<45} {pct('correct'):>10} {pct('incorrect'):>11} {pct('correct_fact_unbracketed'):>9}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gpt", action="store_true",
                        help="Skip GPT-4o-mini annotation (unbracketed model analysis will be skipped)")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    cache = AnnotationCache(CACHE_PATH)

    # ── check OpenAI key ──────────────────────────────────────────────────────
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    if not has_openai and not args.skip_gpt:
        log.warning(
            "OPENAI_API_KEY is not set. "
            "Unbracketed model analysis requires GPT-4o-mini. "
            "Re-run with --skip-gpt to run only the bracketed analysis, "
            "or set OPENAI_API_KEY and re-run."
        )
        sys.exit(1)

    # ── load files ────────────────────────────────────────────────────────────
    log.info("Loading eval files …")
    brack_sft   = load_eval_json(BRACK_DIR  / "sft_lr1.5e-4_epmax30_seed1_global_step_570__on_train_eval_eval.json")
    brack_orig  = load_eval_json(BRACK_DIR  / "sft_lr1.5e-4_epmax30_seed1_global_step_570__on_train_eval_origqa_eval.json")
    unbrack_sft = load_eval_json(UNBRACK_DIR / "sft_lr1.5e-4_epmax30_seed1_global_step_570__on_train_eval_eval.json")
    unbrack_orig = load_eval_json(UNBRACK_DIR / "sft_lr1.5e-4_epmax30_seed1_global_step_570__on_train_eval_origqa_eval.json")

    log.info(f"Bracketed SFT train rows  : {len(brack_sft['rows'])}  origQA rows: {len(brack_orig['rows'])}")
    log.info(f"Unbracketed SFT train rows: {len(unbrack_sft['rows'])}  origQA rows: {len(unbrack_orig['rows'])}")
    log.info(f"Bracketed model origQA accuracy  : {brack_orig['metrics']['accuracy']:.3f}")
    log.info(f"Unbracketed model origQA accuracy: {unbrack_orig['metrics']['accuracy']:.3f}")

    # ── build bags of facts ───────────────────────────────────────────────────

    # Bracketed: extract [..] directly from SFT targets
    log.info("Building bracketed bag of facts …")
    brack_bag_raw: List[str] = []
    for row in brack_sft["rows"]:
        brack_bag_raw.extend(extract_brackets(row["answer"]))
    brack_bag_norm = [normalize(s) for s in brack_bag_raw]
    brack_bag_set  = set(brack_bag_norm)
    log.info(f"  Bracketed bag: {len(brack_bag_raw)} raw spans, {len(brack_bag_set)} unique")

    # Unbracketed: GPT-4o-mini on plain-text SFT targets
    unbrack_bag_set: set = set()
    unbrack_bag_norm: List[str] = []
    if has_openai and not args.skip_gpt:
        log.info("Annotating unbracketed SFT targets with GPT-4o-mini …")
        ub_target_texts = [row["answer"] for row in unbrack_sft["rows"]]
        ub_target_annotated = annotate_texts_gpt(
            ub_target_texts, cache, system_prompt, max_workers=args.max_workers
        )
        unbrack_bag_raw: List[str] = []
        for ann in ub_target_annotated:
            unbrack_bag_raw.extend(extract_brackets(ann))
        unbrack_bag_norm = [normalize(s) for s in unbrack_bag_raw]
        unbrack_bag_set  = set(unbrack_bag_norm)
        log.info(f"  Unbracketed bag: {len(unbrack_bag_raw)} raw spans, {len(unbrack_bag_set)} unique")

    # ── extract spans from origQA responses ──────────────────────────────────

    # Categories for each row
    brack_cats  = [classify_row(r) for r in brack_orig["rows"]]
    unbrack_cats = [classify_row(r) for r in unbrack_orig["rows"]]

    # Bracketed model — primary: [..] brackets
    log.info("Extracting bracketed model spans (primary: [..]) …")
    brack_resp_brackets = [
        extract_brackets((r.get("responses") or [""])[0])
        for r in brack_orig["rows"]
    ]

    # Bracketed model — NER control
    log.info("Extracting bracketed model spans (NER control) …")
    nlp = load_spacy()
    brack_resp_ner = [
        extract_ner_spans((r.get("responses") or [""])[0], nlp)
        for r in brack_orig["rows"]
    ]

    # Unbracketed model — GPT-4o-mini
    unbrack_resp_spans: List[List[str]] = []
    if has_openai and not args.skip_gpt:
        log.info("Annotating unbracketed model responses with GPT-4o-mini …")
        ub_resp_texts = [(r.get("responses") or [""])[0] for r in unbrack_orig["rows"]]
        ub_resp_annotated = annotate_texts_gpt(
            ub_resp_texts, cache, system_prompt, max_workers=args.max_workers
        )
        unbrack_resp_spans = [extract_brackets(ann) for ann in ub_resp_annotated]

    # Unbracketed model — NER control (same nlp)
    log.info("Extracting unbracketed model spans (NER control) …")
    unbrack_resp_ner = [
        extract_ner_spans((r.get("responses") or [""])[0], nlp)
        for r in unbrack_orig["rows"]
    ]

    # ── compute metrics ───────────────────────────────────────────────────────

    all_results = {}

    log.info("Computing metrics …")

    # Config 1: bracketed model, bracket extraction
    m1 = compute_metrics(brack_resp_brackets, brack_cats, brack_bag_set, brack_bag_norm)
    all_results["bracketed_model/bracket_extraction"] = {
        "bag_size": len(brack_bag_set), "metrics": m1
    }

    # Config 2: bracketed model, NER control
    m2 = compute_metrics(brack_resp_ner, brack_cats, brack_bag_set, brack_bag_norm)
    all_results["bracketed_model/ner_control"] = {
        "bag_size": len(brack_bag_set), "metrics": m2
    }

    if has_openai and not args.skip_gpt:
        # Config 3: unbracketed model, GPT-4o-mini extraction
        m3 = compute_metrics(unbrack_resp_spans, unbrack_cats, unbrack_bag_set, unbrack_bag_norm)
        all_results["unbracketed_model/gpt_extraction"] = {
            "bag_size": len(unbrack_bag_set), "metrics": m3
        }

    # Config 4: unbracketed model, NER control
    m4 = compute_metrics(unbrack_resp_ner, unbrack_cats, unbrack_bag_set if unbrack_bag_set else brack_bag_set, unbrack_bag_norm if unbrack_bag_norm else brack_bag_norm)
    all_results["unbracketed_model/ner_control"] = {
        "bag_size": len(unbrack_bag_set) if unbrack_bag_set else len(brack_bag_set),
        "metrics": m4,
        "note": "bag is unbracketed-GPT bag if available, else bracketed bag" if not unbrack_bag_set else "",
    }

    # ── print results ─────────────────────────────────────────────────────────

    print("\n")
    print("=" * 76)
    print("  BRACKETING ABLATION EXPERIMENT")
    print(f"  Bracketed model origQA accuracy  : {brack_orig['metrics']['accuracy']:.3f}  ({brack_orig['metrics']['correct']}/{brack_orig['metrics']['total_responses']})")
    print(f"  Unbracketed model origQA accuracy: {unbrack_orig['metrics']['accuracy']:.3f}  ({unbrack_orig['metrics']['correct']}/{unbrack_orig['metrics']['total_responses']})")
    print("=" * 76)

    for cfg, res in all_results.items():
        print_metrics(cfg, res["metrics"], res["bag_size"])

    print_comparison_table(all_results)

    # ── save results ──────────────────────────────────────────────────────────

    # Convert defaultdict to plain dict for JSON serialisation
    def to_plain(obj):
        if isinstance(obj, dict):
            return {k: to_plain(v) for k, v in obj.items()}
        return obj

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(to_plain(all_results), f, indent=2)
    log.info(f"Results written to {RESULTS_PATH}")
    print(f"\nResults written to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
