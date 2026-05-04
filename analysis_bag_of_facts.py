"""
Analysis: Do bracketed response spans come from the SFT bag of facts?

Bag of facts = bracketed spans in SFT answers.
Test set = bracketed spans in model responses.

This script supports:
1) Single-file mode: analyze one input file via --origqa.
2) Batch mode: analyze all .json eval files in --generations_dir and write
    per-file reports to <generations_dir>/bag_of_facts_analysis/.
"""

import argparse
import json
from pathlib import Path
import re
import string
from collections import defaultdict


# -- helpers -----------------------------------------------------------------

def extract_brackets(text):
    """Return list of spans found inside [...] in text."""
    if text is None:
        return []
    return re.findall(r"\[([^\[\]]+)\]", str(text))


def normalize(span):
    """Lowercase, strip punctuation, sort tokens."""
    span = str(span).lower()
    span = span.translate(str.maketrans("", "", string.punctuation))
    tokens = sorted(span.split())
    return " ".join(tokens)


def in_bag_exact(span, bag_set):
    return normalize(span) in bag_set


def in_bag_fuzzy(span, bag_list, threshold=85):
    try:
        from rapidfuzz import fuzz

        norm = normalize(span)
        for b in bag_list:
            if fuzz.ratio(norm, b) >= threshold:
                return True
    except ImportError:
        pass
    return False


def normalize_simple(text):
    """Lowercase and strip punctuation, preserving word order."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def first_response(value):
    """Extract a single response string from list/array/scalar formats."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    try:
        # Handles list, tuple, and numpy array without hard dependency on numpy.
        if len(value) == 0:
            return ""
        return "" if value[0] is None else str(value[0])
    except Exception:
        return str(value)


def row_value(row, key, default=None):
    """Read value from dict-like rows safely."""
    if isinstance(row, dict):
        return row.get(key, default)
    return default


# -- loading -----------------------------------------------------------------

_DEFAULT_SFT = "data/simpleqa/partition/factual_anchor/smoketest/sft/train.parquet"
_DEFAULT_ORIGQA = (
    "outputs/sft/simpleqa_factual_anchor_sft/"
    "sft_lr1.5e-4_epmax30_seed1/global_step_1290/generations/"
    "sft_lr1.5e-4_epmax30_seed1_global_step_1290__on_train_eval_origqa_eval.json"
)


def load_sft_answers(path):
    """Load SFT answers from JSON or Parquet and return answer list."""
    ext = Path(path).suffix.lower()

    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        rows = data.get("rows", [])
        return [row.get("answer", "") for row in rows], len(rows)

    if ext == ".parquet":
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError(
                "Parquet input requires pandas (and pyarrow/fastparquet). "
                "Install it or pass a JSON file to --sft."
            ) from e

        df = pd.read_parquet(path)
        if "answer" not in df.columns:
            raise KeyError(f'Expected column "answer" in SFT parquet: {path}')

        answers = ["" if x is None else str(x) for x in df["answer"].tolist()]
        return answers, len(answers)

    raise ValueError(f"Unsupported --sft file type: {path}")


def load_rows(path):
    """Load analysis rows from JSON eval output or generation Parquet."""
    ext = Path(path).suffix.lower()

    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        return data.get("rows", [])

    if ext == ".parquet":
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError(
                "Parquet input requires pandas (and pyarrow/fastparquet)."
            ) from e

        df = pd.read_parquet(path)
        return df.to_dict("records")

    raise ValueError(f"Unsupported input file type: {path}")


# -- analysis ----------------------------------------------------------------

def get_category(row):
    """correct | incorrect | not_attempted | unknown"""
    graders = row_value(row, "graders", None)
    if graders:
        try:
            return graders[0].get("evaluation", "unknown")
        except Exception:
            return "unknown"
    return "unknown"


def correct_fact_in_response_unbracketed(row):
    """Check if a gold span appears unbracketed in the response."""
    gold_spans = extract_brackets(row_value(row, "answer", ""))
    if not gold_spans:
        return False

    response = first_response(row_value(row, "responses", ""))
    response_no_brackets = re.sub(r"\[[^\[\]]+\]", " ", response)
    response_norm = normalize_simple(response_no_brackets)

    for gs in gold_spans:
        norm_gs = normalize_simple(gs)
        if norm_gs and norm_gs in response_norm:
            return True
    return False


def analyze_rows(rows, bag_set, bag_normalized):
    """Compute span-level and response-level bag-of-facts statistics."""
    stats = defaultdict(
        lambda: {
            "total": 0,
            "exact_match": 0,
            "fuzzy_match": 0,
            "no_match": 0,
        }
    )
    response_stats = defaultdict(
        lambda: {
            "responses": 0,
            "any_span": 0,
            "all_exact": 0,
            "all_fuzzy": 0,
        }
    )

    span_records = []

    for row in rows:
        response = first_response(row_value(row, "responses", ""))
        cat = get_category(row)

        if cat == "incorrect" and correct_fact_in_response_unbracketed(row):
            cat_display = "correct_fact_unbracketed"
        else:
            cat_display = cat

        resp_spans = extract_brackets(response)
        response_stats[cat_display]["responses"] += 1
        if resp_spans:
            response_stats[cat_display]["any_span"] += 1

        for span in resp_spans:
            exact = in_bag_exact(span, bag_set)
            fuzzy = (not exact) and in_bag_fuzzy(span, bag_normalized)
            stats[cat_display]["total"] += 1
            if exact:
                stats[cat_display]["exact_match"] += 1
            elif fuzzy:
                stats[cat_display]["fuzzy_match"] += 1
            else:
                stats[cat_display]["no_match"] += 1

            span_records.append(
                {
                    "cat": cat_display,
                    "span": span,
                    "exact": exact,
                    "fuzzy": fuzzy,
                }
            )

        if resp_spans and all(in_bag_exact(s, bag_set) for s in resp_spans):
            response_stats[cat_display]["all_exact"] += 1
        if resp_spans and all(
            in_bag_exact(s, bag_set) or in_bag_fuzzy(s, bag_normalized)
            for s in resp_spans
        ):
            response_stats[cat_display]["all_fuzzy"] += 1

    return stats, response_stats, span_records


def format_report(input_path, sft_row_count, bag_raw_count, bag_set, stats, response_stats, span_records):
    """Build a text report matching the previous output style."""
    all_spans = list(span_records)
    total_spans = len(all_spans)
    exact_total = sum(1 for r in all_spans if r["exact"])
    fuzzy_total = sum(1 for r in all_spans if r["fuzzy"])
    match_total = exact_total + fuzzy_total

    out = []
    out.append(f"Input file: {input_path}")
    out.append(f"SFT source rows: {sft_row_count}")
    out.append(
        f"Bag of facts: {bag_raw_count} raw spans, {len(bag_set)} unique (normalized)"
    )
    out.append(f"Sample bag entries (first 10): {sorted(list(bag_set))[:10]}")
    out.append("")

    out.append("=" * 72)
    out.append("OVERALL SPAN-LEVEL RESULTS")
    out.append("=" * 72)
    out.append(f"Total bracketed spans in model responses  : {total_spans}")

    if total_spans > 0:
        out.append(
            f"  Exact in-bag matches                    : {exact_total:4d}  ({exact_total/total_spans*100:.1f}%)"
        )
        out.append(
            f"  Fuzzy in-bag matches (rapidfuzz >=85)   : {fuzzy_total:4d}  ({fuzzy_total/total_spans*100:.1f}%)"
        )
        out.append(
            f"  Combined in-bag                         : {match_total:4d}  ({match_total/total_spans*100:.1f}%)"
        )
        out.append(
            f"  No match                                : {total_spans-match_total:4d}  ({(total_spans-match_total)/total_spans*100:.1f}%)"
        )
    else:
        out.append("  No bracketed spans found in responses.")
    out.append("")

    out.append("=" * 72)
    out.append("STRATIFIED BY RESPONSE CORRECTNESS (span level)")
    out.append("=" * 72)
    header = (
        f"{'Category':<28} {'Spans':>6} {'Exact%':>8} {'Fuzzy%':>8} {'InBag%':>8} {'None%':>8}"
    )
    out.append(header)
    out.append("-" * 72)

    cat_order = [
        "correct",
        "incorrect",
        "correct_fact_unbracketed",
        "not_attempted",
        "unknown",
    ]
    for cat in cat_order:
        if cat not in stats:
            continue
        s = stats[cat]
        t = s["total"]
        if t == 0:
            continue
        e_pct = s["exact_match"] / t * 100
        f_pct = s["fuzzy_match"] / t * 100
        n_pct = s["no_match"] / t * 100
        ib_pct = e_pct + f_pct
        out.append(
            f"{cat:<28} {t:>6} {e_pct:>7.1f}% {f_pct:>7.1f}% {ib_pct:>7.1f}% {n_pct:>7.1f}%"
        )

    out.append("")
    out.append("=" * 72)
    out.append("STRATIFIED BY RESPONSE CORRECTNESS (response level)")
    out.append("=" * 72)
    header2 = (
        f"{'Category':<28} {'Resp':>6} {'HasSpan%':>10} {'AllExact%':>10} {'AllInBag%':>10}"
    )
    out.append(header2)
    out.append("-" * 72)

    for cat in cat_order:
        if cat not in response_stats:
            continue
        rs = response_stats[cat]
        n = rs["responses"]
        if n == 0:
            continue
        hs_pct = rs["any_span"] / n * 100
        ae_pct = rs["all_exact"] / n * 100
        af_pct = rs["all_fuzzy"] / n * 100
        out.append(
            f"{cat:<28} {n:>6} {hs_pct:>9.1f}% {ae_pct:>9.1f}% {af_pct:>9.1f}%"
        )

    out.append("")
    out.append("=" * 72)
    out.append("EXAMPLE NO-MATCH SPANS (first 20)")
    out.append("=" * 72)
    no_match = [r for r in span_records if not r["exact"] and not r["fuzzy"]]
    if no_match:
        for r in no_match[:20]:
            out.append(f"  [{r['cat']:>8}] \"{r['span']}\"")
    else:
        out.append("  None")

    return "\n".join(out) + "\n"


def analyze_file(input_path, sft_row_count, bag_raw_count, bag_set, bag_normalized, output_path=None):
    rows = load_rows(input_path)
    stats, response_stats, span_records = analyze_rows(rows, bag_set, bag_normalized)
    report = format_report(
        input_path,
        sft_row_count,
        bag_raw_count,
        bag_set,
        stats,
        response_stats,
        span_records,
    )

    if output_path is None:
        print(report, end="")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Wrote: {output_path}")


# -- cli ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default=_DEFAULT_SFT)
    parser.add_argument("--origqa", default=_DEFAULT_ORIGQA)
    parser.add_argument(
        "--generations_dir",
        default=None,
        help=(
            "Directory containing eval JSON files. If set, the script "
            "analyzes every '*_eval.json' in this directory and writes reports to "
            "<generations_dir>/bag_of_facts_analysis/."
        ),
    )
    args = parser.parse_args()

    sft_answers, sft_row_count = load_sft_answers(args.sft)
    bag_raw = []
    for answer in sft_answers:
        bag_raw.extend(extract_brackets(answer))

    bag_normalized = [normalize(s) for s in bag_raw]
    bag_set = set(bag_normalized)

    if args.generations_dir:
        gen_dir = Path(args.generations_dir)
        if not gen_dir.is_dir():
            raise NotADirectoryError(f"--generations_dir is not a directory: {gen_dir}")

        json_files = sorted(gen_dir.glob("*_eval.json"))
        if not json_files:
            # Fallback to all JSON files when *_eval.json naming is not used.
            json_files = sorted(gen_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in: {gen_dir}")

        out_dir = gen_dir / "bag_of_facts_analysis"
        for jf in json_files:
            out_path = out_dir / f"{jf.stem}.txt"
            analyze_file(
                str(jf),
                sft_row_count,
                len(bag_raw),
                bag_set,
                bag_normalized,
                output_path=out_path,
            )
        print(f"Done. Reports written to: {out_dir}")
    else:
        analyze_file(
            args.origqa,
            sft_row_count,
            len(bag_raw),
            bag_set,
            bag_normalized,
            output_path=None,
        )


if __name__ == "__main__":
    main()
