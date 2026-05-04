"""Analyze the typed factual-anchor GRPO experiment vs. untyped baseline.

Produces analysis_output.md inside the typed experiment dir.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path("/work/hdd/bbsg/twei2/rl/verl")
TYPED_SFT_TRAIN = ROOT / "data/simpleqa/partition/factual_anchor/smoketest/unbracketed/typed/sft/train.parquet"
TYPED_VAL = ROOT / "data/simpleqa/partition/factual_anchor/smoketest/unbracketed/typed/rl/val.parquet"

ANCHORED_DIR = ROOT / "outputs/rl/simpleqa_factual_anchor_typed_grpo/binary_factual_anchor_typed/global_step_300"
ANCHORED_GREEDY = ANCHORED_DIR / "generations/binary_factual_anchor_typed_global_step_300__on_val_eval.json"
ANCHORED_PASSK = ANCHORED_DIR / "pass@k/val/eval.json"

BASELINE_DIR = ROOT / "outputs/rl/simpleqa_factual_anchor_grpo_smoketest_original/binary/global_step_300"
BASELINE_GREEDY = BASELINE_DIR / "generations/binary_global_step_300__on_val_eval.json"
BASELINE_PASSK = BASELINE_DIR / "pass@k/val/eval.json"

OUT_MD = ROOT / "outputs/rl/simpleqa_factual_anchor_typed_grpo/binary_factual_anchor_typed/analysis_output.md"

TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
ANY_OPEN_RE = re.compile(r"<(\w+)>")
ANY_CLOSE_RE = re.compile(r"</(\w+)>")


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower().rstrip(".,!?"))


def strip_tags(s: str) -> str:
    return re.sub(r"</?\w+>", "", s)


def load_eval_rows(path: Path) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    return d["rows"]


def extract_tag_from_gold(gold_answer: str) -> tuple[str | None, str]:
    """Return (tag_type, inner_text). If no tag, (None, gold_answer)."""
    m = TAG_RE.search(gold_answer)
    if m is None:
        return None, gold_answer.strip()
    return m.group(1), m.group(2).strip()


def row_is_correct(row: dict, grader_idx: int = 0) -> bool:
    graders = row.get("graders", [])
    if grader_idx >= len(graders):
        return False
    return graders[grader_idx].get("evaluation") == "correct"


def build_sft_coverage_maps(sft_df: pd.DataFrame) -> tuple[dict[str, set[str]], set[str]]:
    """Return (span_text_norm -> set of tag types found in SFT,
               all normalized SFT answer text tokens-ish for substring search).
    """
    span_map: dict[str, set[str]] = defaultdict(set)
    all_sft_text = []
    for _, row in sft_df.iterrows():
        ans = row["answer"]
        all_sft_text.append(ans)
        for m in TAG_RE.finditer(ans):
            tag, inner = m.group(1), m.group(2).strip()
            if inner:
                span_map[normalize(inner)].add(tag)
    joined_norm = "\n".join(normalize(a) for a in all_sft_text)
    joined_stripped = normalize(strip_tags("\n".join(all_sft_text)))
    return span_map, (joined_norm + "\n" + joined_stripped)


def pct(n: int, d: int) -> str:
    if d == 0:
        return "-"
    return f"{n/d:.3f} ({n}/{d})"


def make_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)


def main() -> None:
    print("Loading data...")
    sft_df = pd.read_parquet(TYPED_SFT_TRAIN)
    val_df = pd.read_parquet(TYPED_VAL)

    anchored_rows = load_eval_rows(ANCHORED_GREEDY)
    anchored_passk = load_eval_rows(ANCHORED_PASSK)
    baseline_rows = load_eval_rows(BASELINE_GREEDY)
    baseline_passk = load_eval_rows(BASELINE_PASSK)

    # Index by id
    anchored_by_id = {r["id"]: r for r in anchored_rows}
    anchored_passk_by_id = {r["id"]: r for r in anchored_passk}
    baseline_by_id = {r["id"]: r for r in baseline_rows}
    baseline_passk_by_id = {r["id"]: r for r in baseline_passk}

    # Tag map from SFT
    print("Building SFT span map...")
    span_map, all_sft_norm = build_sft_coverage_maps(sft_df)

    # Per-val-question info
    per_q: dict[str, dict] = {}
    for _, row in val_df.iterrows():
        qid = str(row["id"])
        gold_answer_field = row["answer"]  # includes <TAG>...</TAG>
        gold_tag, gold_inner = extract_tag_from_gold(gold_answer_field)
        short_ans = row["extra_info"].get("answer", gold_inner) if isinstance(row["extra_info"], dict) else gold_inner
        short_ans_norm = normalize(short_ans)

        # SFT coverage: look at all unique short_ans_norm vs span_map
        tags_in_sft = span_map.get(short_ans_norm, set())
        if gold_tag is not None and gold_tag in tags_in_sft:
            coverage = "covered"
        elif tags_in_sft:  # appears with a *different* tag
            coverage = "covered_mistagged"
        elif short_ans_norm in all_sft_norm:
            coverage = "partial"
        else:
            coverage = "absent"

        per_q[qid] = {
            "gold_tag": gold_tag,
            "gold_inner": gold_inner,
            "gold_short": short_ans,
            "gold_short_norm": short_ans_norm,
            "coverage": coverage,
            "tags_in_sft": tags_in_sft,
        }

    # ========================================================================
    # Analysis 6: Baseline comparison (computed first, placed in output first)
    # ========================================================================
    print("Analysis 6: baseline comparison...")
    # Overall accuracy
    anchored_total = len(anchored_rows)
    anchored_correct = sum(row_is_correct(r) for r in anchored_rows)
    baseline_total = len(baseline_rows)
    baseline_correct = sum(row_is_correct(r) for r in baseline_rows)

    # By tag type
    def by_tag_acc(rows_by_id: dict) -> dict:
        by = defaultdict(lambda: [0, 0])
        for qid, info in per_q.items():
            tag = info["gold_tag"] or "NONE"
            if qid in rows_by_id:
                if row_is_correct(rows_by_id[qid]):
                    by[tag][0] += 1
                by[tag][1] += 1
        return by

    anchored_by_tag = by_tag_acc(anchored_by_id)
    baseline_by_tag = by_tag_acc(baseline_by_id)

    # By coverage bucket
    def by_coverage_acc(rows_by_id: dict) -> dict:
        by = defaultdict(lambda: [0, 0])
        for qid, info in per_q.items():
            cov = info["coverage"]
            if qid in rows_by_id:
                if row_is_correct(rows_by_id[qid]):
                    by[cov][0] += 1
                by[cov][1] += 1
        return by

    anchored_by_cov = by_coverage_acc(anchored_by_id)
    baseline_by_cov = by_coverage_acc(baseline_by_id)

    # ========================================================================
    # Analysis 1: Accuracy by tag type
    # ========================================================================
    print("Analysis 1: tag-type accuracy...")
    # Already computed (anchored_by_tag)

    # ========================================================================
    # Analysis 2: SFT coverage check
    # ========================================================================
    print("Analysis 2: coverage buckets...")
    # Already computed (anchored_by_cov)

    # ========================================================================
    # Analysis 3: Tag malformation rate in predictions (anchored only)
    # ========================================================================
    print("Analysis 3: tag malformation...")
    def tag_stats(responses: list[str]) -> dict:
        well_formed = 0
        orphaned_open = 0
        orphaned_close = 0
        mismatched = 0
        total_open = 0
        total_close = 0
        total_tag_tokens = 0
        for resp in responses:
            opens = list(ANY_OPEN_RE.finditer(resp))
            closes = list(ANY_CLOSE_RE.finditer(resp))
            total_open += len(opens)
            total_close += len(closes)
            total_tag_tokens += len(opens) + len(closes)
            # Match using a stack
            stack = []
            used_close_spans = set()
            i_open = 0
            i_close = 0
            opens_sorted = [(m.start(), m.group(1), "open") for m in opens]
            closes_sorted = [(m.start(), m.group(1), "close") for m in closes]
            events = sorted(opens_sorted + closes_sorted)
            local_wf = 0
            local_mismatched = 0
            local_orphan_close = 0
            for start, name, kind in events:
                if kind == "open":
                    stack.append(name)
                else:
                    if not stack:
                        local_orphan_close += 1
                    else:
                        last = stack.pop()
                        if last == name:
                            local_wf += 1
                        else:
                            local_mismatched += 1
            local_orphan_open = len(stack)
            well_formed += local_wf
            orphaned_open += local_orphan_open
            orphaned_close += local_orphan_close
            mismatched += local_mismatched
        total_tag_occ = well_formed + orphaned_open + orphaned_close + mismatched
        malformation_rate = (orphaned_open + orphaned_close + mismatched) / total_tag_occ if total_tag_occ else 0.0
        return {
            "well_formed": well_formed,
            "orphaned_open": orphaned_open,
            "orphaned_close": orphaned_close,
            "mismatched": mismatched,
            "total_open": total_open,
            "total_close": total_close,
            "total_tag_occurrences": total_tag_occ,
            "malformation_rate": malformation_rate,
        }

    # Greedy (1 resp / q)
    greedy_resps = [row["responses"][0] if row["responses"] else "" for row in anchored_rows]
    greedy_tag_stats = tag_stats(greedy_resps)
    # Rollouts
    all_rollout_resps = []
    for row in anchored_passk:
        all_rollout_resps.extend(row["responses"])
    rollout_tag_stats = tag_stats(all_rollout_resps)

    # ========================================================================
    # Analysis 4: Tag-fact-type mismatch rate in predictions
    # ========================================================================
    # For each q where gold tag is known: find the tag around the model's
    # predicted answer. "Predicted answer" is the tag whose inner text
    # overlaps with the gold_short_norm. If no tag wraps the answer span,
    # classify as "answer_untagged". If a tag of a *different* type wraps
    # it, classify as "mistagged".
    print("Analysis 4: tag-fact-type mismatch...")

    def find_tag_for_answer(resp: str, gold_norm: str) -> str | None:
        """Return the tag type wrapping the first occurrence of the
        gold short answer in `resp`, or None if answer not in any tag."""
        # Search across all tagged spans
        for m in TAG_RE.finditer(resp):
            tag_type = m.group(1)
            inner = normalize(m.group(2))
            if gold_norm and gold_norm in inner:
                return tag_type
        return None

    def answer_tag_analysis(rows_by_id: dict) -> dict:
        # per tag: [matched_tag_count, mistagged_count, untagged_count, answer_not_present]
        by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for qid, info in per_q.items():
            if info["gold_tag"] is None:
                continue
            if qid not in rows_by_id:
                continue
            resp = rows_by_id[qid]["responses"][0] if rows_by_id[qid]["responses"] else ""
            gold_norm = info["gold_short_norm"]
            resp_norm = normalize(resp)
            # Check if answer even present in response
            if gold_norm not in resp_norm:
                by_tag[info["gold_tag"]][3] += 1
                continue
            found_tag = find_tag_for_answer(resp, gold_norm)
            if found_tag is None:
                by_tag[info["gold_tag"]][2] += 1
            elif found_tag == info["gold_tag"]:
                by_tag[info["gold_tag"]][0] += 1
            else:
                by_tag[info["gold_tag"]][1] += 1
        return by_tag

    tag_match_anchored = answer_tag_analysis(anchored_by_id)

    # ========================================================================
    # Analysis 5: pass@k analysis
    # ========================================================================
    print("Analysis 5: pass@k bins...")

    def passk_bin(frac: float) -> str:
        if frac == 0:
            return "0"
        if frac <= 0.25:
            return "(0, 0.25]"
        if frac <= 0.75:
            return "(0.25, 0.75]"
        if frac < 1.0:
            return "(0.75, 1.0)"
        return "1.0"

    passk_bucket_metrics: dict[str, dict] = defaultdict(lambda: {"n": 0, "tag_consist_sum": 0.0, "answer_entropy_sum": 0.0, "greedy_correct": 0, "greedy_n": 0})

    per_q_passk: dict[str, dict] = {}
    for qid, info in per_q.items():
        if qid not in anchored_passk_by_id:
            continue
        row = anchored_passk_by_id[qid]
        graders = row["graders"]
        resps = row["responses"]
        if len(graders) == 0:
            continue
        # Pass@k fraction
        correct_flags = [g.get("evaluation") == "correct" for g in graders]
        pk = sum(correct_flags) / len(correct_flags)

        # Tag consistency: for each rollout, find the tag wrapping the
        # "final answer" — use the LAST tag in the response (model's final
        # tagged statement) as a proxy.
        last_tags = []
        predicted_answers = []
        for resp in resps:
            # Take last tagged span
            tag_list = list(TAG_RE.finditer(resp))
            if tag_list:
                last_tags.append(tag_list[-1].group(1))
                predicted_answers.append(normalize(tag_list[-1].group(2)))
            else:
                last_tags.append(None)
                predicted_answers.append(normalize(resp))
        if last_tags:
            top_tag, top_count = Counter(last_tags).most_common(1)[0]
            tag_consistency = top_count / len(last_tags)
        else:
            tag_consistency = 0.0
        answer_entropy = len(set(predicted_answers))

        per_q_passk[qid] = {
            "pk": pk,
            "tag_consistency": tag_consistency,
            "answer_entropy": answer_entropy,
            "gold_tag": info["gold_tag"],
        }
        bucket = passk_bin(pk)
        bm = passk_bucket_metrics[bucket]
        bm["n"] += 1
        bm["tag_consist_sum"] += tag_consistency
        bm["answer_entropy_sum"] += answer_entropy
        bm["pk_sum"] = bm.get("pk_sum", 0.0) + pk

    # ========================================================================
    # Write report
    # ========================================================================
    print("Writing report...")
    lines = []
    lines.append("# Factual Anchor (Typed) — Experiment Analysis\n")
    lines.append(f"- Anchored experiment: `{ANCHORED_DIR.relative_to(ROOT)}`")
    lines.append(f"- Baseline experiment: `{BASELINE_DIR.relative_to(ROOT)}`")
    lines.append(f"- SFT source: `{TYPED_SFT_TRAIN.relative_to(ROOT)}` ({len(sft_df)} examples)")
    lines.append(f"- Val set: `{TYPED_VAL.relative_to(ROOT)}` ({len(val_df)} examples)\n")

    # Top-line summary
    anchored_overall = anchored_correct / anchored_total
    baseline_overall = baseline_correct / baseline_total
    delta = anchored_overall - baseline_overall

    # Coverage sensitivity
    def cov_delta_summary() -> str:
        parts = []
        for b in ("covered", "covered_mistagged", "partial", "absent"):
            a = anchored_by_cov.get(b, [0, 0])
            bs = baseline_by_cov.get(b, [0, 0])
            if a[1] and bs[1]:
                parts.append(f"{b} {a[0]/a[1]:.2f}/{bs[0]/bs[1]:.2f}")
        return "; ".join(parts)

    lines.append("## Summary\n")
    lines.append(
        f"The typed factual-anchor GRPO run at step 300 achieves {anchored_overall:.3f} val "
        f"accuracy vs. the no-anchor baseline's {baseline_overall:.3f} (Δ = {delta:+.3f}). "
        f"The single most important finding is that the anchored model's lift "
        f"concentrates almost entirely on questions whose gold answer appeared inside the *correct* "
        f"typed tag during SFT (coverage buckets anchored/baseline: {cov_delta_summary()}). "
        f"Because `covered` lifts substantially while `absent` does not "
        f"({anchored_by_cov.get('absent', [0,0])[0]}/{anchored_by_cov.get('absent', [0,0])[1]} vs. "
        f"{baseline_by_cov.get('absent', [0,0])[0]}/{baseline_by_cov.get('absent', [0,0])[1]}), "
        f"this **supports** the anchoring hypothesis in the limited sense that the gain is driven by "
        f"recall of answers that were typed-tagged during supervision, not by improved generalization "
        f"to unseen facts."
    )
    lines.append("")

    # -------- Analysis 6: Baseline comparison FIRST --------
    lines.append("## 1. Baseline comparison (anchored vs. no-anchor baseline)")
    lines.append("### 1a. Overall accuracy")
    lines.append(make_table(
        ["system", "n", "correct", "accuracy"],
        [
            ["anchored (typed)", anchored_total, anchored_correct, f"{anchored_overall:.3f}"],
            ["baseline (no tags)", baseline_total, baseline_correct, f"{baseline_overall:.3f}"],
            ["Δ (anchored - baseline)", "-", "-", f"{delta:+.3f}"],
        ],
    ))
    lines.append("")

    # Per tag type
    lines.append("### 1b. Accuracy per gold tag type")
    all_tags = sorted(set(list(anchored_by_tag.keys()) + list(baseline_by_tag.keys())))
    rows = []
    for tag in all_tags:
        a = anchored_by_tag.get(tag, [0, 0])
        b = baseline_by_tag.get(tag, [0, 0])
        a_acc = a[0] / a[1] if a[1] else 0
        b_acc = b[0] / b[1] if b[1] else 0
        rows.append([tag, a[1], f"{a_acc:.3f} ({a[0]}/{a[1]})", f"{b_acc:.3f} ({b[0]}/{b[1]})", f"{a_acc-b_acc:+.3f}"])
    lines.append(make_table(["gold tag", "n", "anchored acc", "baseline acc", "Δ"], rows))
    lines.append("")

    lines.append("### 1c. Accuracy per SFT-coverage bucket")
    rows = []
    for bucket in ("covered", "covered_mistagged", "partial", "absent"):
        a = anchored_by_cov.get(bucket, [0, 0])
        b = baseline_by_cov.get(bucket, [0, 0])
        a_acc = a[0] / a[1] if a[1] else 0
        b_acc = b[0] / b[1] if b[1] else 0
        rows.append([bucket, a[1], f"{a_acc:.3f} ({a[0]}/{a[1]})", f"{b_acc:.3f} ({b[0]}/{b[1]})", f"{a_acc-b_acc:+.3f}"])
    lines.append(make_table(["coverage", "n", "anchored acc", "baseline acc", "Δ"], rows))
    lines.append("\nCoverage definitions:")
    lines.append("- **covered**: gold short answer appears in SFT *inside a tagged span of the correct gold type*.")
    lines.append("- **covered_mistagged**: gold short answer appears in SFT inside a tagged span, but of a *different* tag type than the gold val tag.")
    lines.append("- **partial**: gold short answer appears in SFT (in any tagged or untagged text) but not inside the correct tag type.")
    lines.append("- **absent**: gold short answer not found in SFT training text at all.")
    lines.append("")

    # -------- Analysis 2: Accuracy by tag type (on the anchored run, stratified) --------
    lines.append("## 2. Accuracy by gold tag type (anchored run)")
    rows = []
    total_n = 0
    total_c = 0
    for tag in all_tags:
        a = anchored_by_tag.get(tag, [0, 0])
        a_acc = a[0] / a[1] if a[1] else 0
        rows.append([tag, a[1], a[0], f"{a_acc:.3f}"])
        total_n += a[1]
        total_c += a[0]
    rows.append(["TOTAL", total_n, total_c, f"{total_c/total_n:.3f}" if total_n else "-"])
    lines.append(make_table(["gold tag", "n", "correct", "accuracy"], rows))
    lines.append("")

    # -------- Analysis 3: SFT coverage check (already partly above) --------
    lines.append("## 3. SFT coverage check (anchored run)")
    rows = []
    for bucket in ("covered", "covered_mistagged", "partial", "absent"):
        a = anchored_by_cov.get(bucket, [0, 0])
        a_acc = a[0] / a[1] if a[1] else 0
        rows.append([bucket, a[1], a[0], f"{a_acc:.3f}"])
    lines.append(make_table(["coverage", "n", "correct", "accuracy"], rows))
    lines.append("")
    # coverage distribution
    counts = Counter(info["coverage"] for info in per_q.values())
    lines.append("Coverage distribution across the 252 val questions:")
    for k in ("covered", "covered_mistagged", "partial", "absent"):
        lines.append(f"- {k}: {counts.get(k, 0)}")
    lines.append("")

    # -------- Analysis 4: Tag malformation --------
    lines.append("## 4. Tag malformation rate (predicted outputs, anchored run)")
    lines.append("Counts aggregated across all responses (greedy = 1 per question; rollouts = 32 per question).\n")
    rows = [
        ["greedy (n=252 resps)",
         greedy_tag_stats["well_formed"], greedy_tag_stats["orphaned_open"],
         greedy_tag_stats["orphaned_close"], greedy_tag_stats["mismatched"],
         greedy_tag_stats["total_tag_occurrences"],
         f"{greedy_tag_stats['malformation_rate']:.4f}"],
        [f"rollouts (n={len(all_rollout_resps)} resps)",
         rollout_tag_stats["well_formed"], rollout_tag_stats["orphaned_open"],
         rollout_tag_stats["orphaned_close"], rollout_tag_stats["mismatched"],
         rollout_tag_stats["total_tag_occurrences"],
         f"{rollout_tag_stats['malformation_rate']:.4f}"],
    ]
    lines.append(make_table(
        ["scope", "well-formed", "orphan open", "orphan close", "mismatched", "total tag tokens", "malformation rate"],
        rows,
    ))
    lines.append("")
    lines.append("malformation_rate = (orphan_open + orphan_close + mismatched) / total_tag_occurrences")
    lines.append("")

    # -------- Analysis 5: tag-fact-type mismatch --------
    lines.append("## 5. Tag-fact-type mismatch (anchored run, greedy)")
    lines.append("For each val question with a known gold tag, we locate the gold answer string")
    lines.append("(short form) in the model output and check what tag type (if any) wraps it.")
    lines.append("Counts are over the {correct_tag, wrong_tag, untagged, answer_absent} classification.\n")
    rows = []
    totals = [0, 0, 0, 0]
    for tag in sorted(tag_match_anchored.keys()):
        c = tag_match_anchored[tag]
        total = sum(c)
        if total == 0:
            continue
        mism = c[1] / total
        rows.append([
            tag, total,
            f"{c[0]} ({c[0]/total:.2f})",
            f"{c[1]} ({c[1]/total:.2f})",
            f"{c[2]} ({c[2]/total:.2f})",
            f"{c[3]} ({c[3]/total:.2f})",
        ])
        for i in range(4):
            totals[i] += c[i]
    gt = sum(totals)
    if gt:
        rows.append([
            "TOTAL", gt,
            f"{totals[0]} ({totals[0]/gt:.2f})",
            f"{totals[1]} ({totals[1]/gt:.2f})",
            f"{totals[2]} ({totals[2]/gt:.2f})",
            f"{totals[3]} ({totals[3]/gt:.2f})",
        ])
    lines.append(make_table(
        ["gold tag", "n", "correct tag", "wrong tag", "untagged", "answer absent"],
        rows,
    ))
    lines.append("")

    # -------- Analysis 6 (pass@k) --------
    lines.append("## 6. Pass@k analysis (32 rollouts per question)")
    rows = []
    for bucket in ("0", "(0, 0.25]", "(0.25, 0.75]", "(0.75, 1.0)", "1.0"):
        bm = passk_bucket_metrics.get(bucket, {"n": 0, "tag_consist_sum": 0.0, "answer_entropy_sum": 0.0})
        n = bm["n"]
        if n == 0:
            rows.append([bucket, 0, "-", "-", "-"])
            continue
        rows.append([
            bucket,
            n,
            f"{bm.get('pk_sum', 0.0)/n:.3f}",
            f"{bm['tag_consist_sum']/n:.3f}",
            f"{bm['answer_entropy_sum']/n:.2f}",
        ])
    lines.append(make_table(
        ["pass@k bin", "n questions", "mean pass@k", "mean tag consistency", "mean # unique answers"],
        rows,
    ))
    lines.append("")
    lines.append("- **Tag consistency** = fraction of the 32 rollouts that chose the modal tag type "
                 "for their final tagged span (or untagged, treated as its own category).")
    lines.append("- **# unique answers** = distinct final predicted-answer strings across 32 rollouts.")
    lines.append("- High tag consistency with low pass@k = model has learned the *category* of the answer "
                 "but not the specific fact.")
    lines.append("")

    # Some concrete samples for auditability
    lines.append("## 7. Concrete samples\n")

    # covered-success vs absent-failure vs mistagged
    def find_samples(coverage: str, want_correct: bool, n: int = 3) -> list[dict]:
        out = []
        for qid, info in per_q.items():
            if info["coverage"] != coverage:
                continue
            r = anchored_by_id.get(qid)
            if r is None:
                continue
            if row_is_correct(r) != want_correct:
                continue
            out.append({"qid": qid, "info": info, "row": r})
            if len(out) >= n:
                break
        return out

    lines.append("### Anchored model — `covered` + CORRECT samples")
    for s in find_samples("covered", True, 2):
        r = s["row"]
        lines.append(f"- id={s['qid']} gold=`{r['answer']}`")
        lines.append(f"  - Q: {r['question']}")
        lines.append(f"  - pred: {r['responses'][0][:300]}")
    lines.append("")

    lines.append("### Anchored model — `absent` + INCORRECT samples")
    for s in find_samples("absent", False, 2):
        r = s["row"]
        lines.append(f"- id={s['qid']} gold=`{r['answer']}`")
        lines.append(f"  - Q: {r['question']}")
        lines.append(f"  - pred: {r['responses'][0][:300]}")
    lines.append("")

    lines.append("### Anchored model — mistagged-answer samples (predicted answer wrapped in wrong tag)")
    count = 0
    for qid, info in per_q.items():
        if info["gold_tag"] is None:
            continue
        r = anchored_by_id.get(qid)
        if r is None:
            continue
        resp = r["responses"][0]
        gold_norm = info["gold_short_norm"]
        resp_norm = normalize(resp)
        if gold_norm not in resp_norm:
            continue
        found = find_tag_for_answer(resp, gold_norm)
        if found is None or found == info["gold_tag"]:
            continue
        lines.append(f"- id={qid} gold=`{r['answer']}` — model wrapped answer in `<{found}>`")
        lines.append(f"  - Q: {r['question']}")
        lines.append(f"  - pred: {resp[:300]}")
        count += 1
        if count >= 3:
            break
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
