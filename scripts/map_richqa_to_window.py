"""Map each rich_qa sample id to the wikipedia window that generated it.

Because the rewriter runs in a ThreadPoolExecutor, ``rich_qa.log`` is heavily
interleaved. The approach here uses three signals:

- ``[sample] seed=<seed>`` as a per-sample lower-bound log-position anchor
  (unique per parquet row via the ``extra_info.augmentation.seed`` field).
- ``[rich_qa] raw_response=<body>`` whose rich_question matches the parquet's
  stored rich_question — upper-bound log-position anchor for the same sample.
- The rich_answer (from the parquet row) is derived from the correct window,
  so the window body should contain the overwhelming majority of the
  rich_answer's non-trivial tokens. This turns out to be the single strongest
  content signal (empirically, the correct window tops the rich_answer token
  overlap ranking 97%+ of the time).

Assignment is greedy over samples sorted by rich_qa position. For each sample
we score every unclaimed window in its (seed_pos, rich_qa_pos] range by:

  score = (
    rich_answer token overlap,   # primary
    url slug token coverage,     # tiebreaker A
    1 if original_answer in window else 0,  # tiebreaker B
    log-position proximity to rich_qa_pos    # tiebreaker C
  )

Claim the highest-scoring window globally. Fallbacks handle rows with no
in-range candidates (small residual).
"""

from __future__ import annotations

import bisect
import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import pandas as pd

LOG_PATH = Path("rich_qa.log")
PARQUET_PATH = Path("data/simpleqa/partition/rich_qa/sft/train.parquet")
OUTPUT_PATH = Path("data/simpleqa/partition/rich_qa/sft/train_windows.parquet")

ENTRY_START_RE = re.compile(r"^\[(sample|extract_window|rich_qa|verify)\]|^=+$")
TOK_RE = re.compile(r"[A-Za-z0-9]{3,}")


def parse_entries(log_text: str) -> list[str]:
    entries: list[str] = []
    current: list[str] = []
    for line in log_text.split("\n"):
        if ENTRY_START_RE.match(line):
            if current:
                entries.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append("\n".join(current))
    return entries


def extract_rich_question(body: str) -> Optional[str]:
    b = body.strip()
    for fence in ("```json", "```"):
        if b.startswith(fence):
            b = b[len(fence):].strip()
            if b.endswith("```"):
                b = b[:-3].strip()
            break
    try:
        obj = json.loads(b)
        rq = obj.get("rich_question")
        if isinstance(rq, str):
            return rq.strip()
    except json.JSONDecodeError:
        pass
    m = re.search(r'"rich_question"\s*:\s*"((?:[^"\\]|\\.)*)"', body, re.DOTALL)
    if m:
        try:
            return m.group(1).encode().decode("unicode_escape").strip()
        except UnicodeDecodeError:
            return m.group(1).strip()
    return None


def url_slug_tokens(url: str) -> list[str]:
    if "/wiki/" not in url:
        return []
    slug = url.split("/wiki/")[-1].split("#")[0].split("?")[0]
    try:
        slug = unquote(slug)
    except Exception:
        pass
    slug = slug.replace("_", " ")
    return [t.lower() for t in re.findall(r"[A-Za-z]+", slug) if len(t) >= 3]


def tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOK_RE.finditer(text or "")}


# Stopwords dropped from rich_answer token set — they're too generic to
# discriminate between candidate windows and their presence in a window is
# not a meaningful match signal.
_STOP = frozenset(
    [
        "the","and","for","with","from","this","that","were","was","are","but",
        "not","his","her","him","she","you","all","any","has","had","have","who",
        "which","when","where","why","how","been","into","their","them","they",
        "its","also","will","would","could","should","about","per","than","then",
        "only","some","most","other","such","over","under","during","between",
        "these","those","while","more","less","one","two","three","four","five",
        "six","seven","eight","nine","ten","new","old","out","after","before",
        "each","many","much","every","both","either","neither","within","without",
        "year","years","day","days","month","months","including","include",
        "among","made","used","using","use","name","named","said","says","known",
        "like","due","first","second","third","fourth","fifth","sixth","seventh",
        "eighth","ninth","tenth","last","next","top","bottom","main","original",
        "rich","answer","question","wikipedia","passage","award","awards","list",
        "full","text","part","parts","section","multiple","given","different",
        "various","several","total","overall","final","final","early","late",
    ]
)


def main() -> None:
    print(f"reading log: {LOG_PATH}")
    log_text = LOG_PATH.read_text(encoding="utf-8")
    entries = parse_entries(log_text)
    print(f"parsed {len(entries)} atomic entries")

    seed_at: dict[int, int] = {}
    window_positions: list[int] = []
    window_bodies: list[str] = []
    window_token_sets: list[set[str]] = []
    rich_qa_rq_at: dict[int, str] = {}

    for i, e in enumerate(entries):
        if e.startswith("[sample] window_preview="):
            body = e[len("[sample] window_preview="):]
            window_positions.append(i)
            window_bodies.append(body)
            window_token_sets.append(tokenize(body))
        elif e.startswith("[rich_qa] raw_response="):
            body = e[len("[rich_qa] raw_response="):]
            rq = extract_rich_question(body)
            if rq:
                rich_qa_rq_at[i] = rq
        elif e.startswith("[sample] seed="):
            try:
                s = int(e.split("=", 1)[1])
            except ValueError:
                continue
            seed_at[i] = s

    print(f"seed entries: {len(seed_at)}")
    print(f"window_preview entries: {len(window_positions)}")
    print(f"rich_qa entries with rich_q: {len(rich_qa_rq_at)}")

    print(f"reading parquet: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)
    rq_to_id: dict[str, str] = {}
    seed_to_id: dict[int, str] = {}
    id_to_source_url: dict[str, str] = {}
    id_to_orig_ans: dict[str, str] = {}
    id_to_seed: dict[str, int] = {}
    id_to_rich_a_tokens: dict[str, set[str]] = {}
    id_to_slug_tokens: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        sid = row["id"]
        rq = row["question"].strip()
        rq_to_id[rq] = sid
        seed = int(row["extra_info"]["augmentation"]["seed"])
        seed_to_id[seed] = sid
        id_to_seed[sid] = seed
        url = row["extra_info"]["augmentation"]["params"].get("source_url", "")
        id_to_source_url[sid] = url
        id_to_orig_ans[sid] = row["extra_info"]["original_answer"]
        rich_ans = row["answer"] or ""
        ra_toks = {t for t in tokenize(rich_ans) if t not in _STOP}
        id_to_rich_a_tokens[sid] = ra_toks
        id_to_slug_tokens[sid] = set(url_slug_tokens(url))
    print(f"parquet rows: {len(df)}")

    seed_pos_by_id: dict[str, int] = {}
    for pos, s in seed_at.items():
        sid = seed_to_id.get(s)
        if sid is not None:
            seed_pos_by_id[sid] = pos
    rich_qa_pos_by_id: dict[str, int] = {}
    for pos, rq in rich_qa_rq_at.items():
        sid = rq_to_id.get(rq)
        if sid is not None:
            rich_qa_pos_by_id[sid] = pos

    anchored = [sid for sid in df["id"] if sid in seed_pos_by_id and sid in rich_qa_pos_by_id]
    print(f"parquet rows with both seed and rich_qa anchors: {len(anchored)}/{len(df)}")

    def windows_in_range(lo_pos: int, hi_pos: int) -> range:
        lo_i = bisect.bisect_right(window_positions, lo_pos)
        hi_i = bisect.bisect_left(window_positions, hi_pos) + 1
        return range(lo_i, min(hi_i, len(window_positions)))

    # Greedy assignment: iterate samples by rich_qa_pos ascending; each sample
    # claims its highest-scoring unclaimed window in-range.
    window_claimed = [False] * len(window_positions)
    assignments: dict[str, tuple[int, str]] = {}
    unresolved: list[str] = []
    ordered = sorted(anchored, key=lambda s: rich_qa_pos_by_id[s])

    score_hist = {"ra_ge_0.5": 0, "ra_ge_0.3": 0, "ra_ge_0.1": 0, "ra_lt_0.1": 0}
    top_ra_scores: list[float] = []

    for sid in ordered:
        lo = seed_pos_by_id[sid]
        hi = rich_qa_pos_by_id[sid]
        ra_toks = id_to_rich_a_tokens[sid]
        ra_n = max(1, len(ra_toks))
        slug = id_to_slug_tokens[sid]
        slug_n = max(1, len(slug))
        ans_lower = (id_to_orig_ans[sid] or "").lower().strip()
        hi_ref = hi

        best_wi = -1
        best_score = (-1.0, -1.0, -1, -1)
        for wi in windows_in_range(lo, hi):
            if window_claimed[wi]:
                continue
            wtoks = window_token_sets[wi]
            ra_overlap = len(ra_toks & wtoks) / ra_n if ra_toks else 0.0
            slug_overlap = len(slug & wtoks) / slug_n if slug else 0.0
            ans_in = 1 if ans_lower and ans_lower in window_bodies[wi].lower() else 0
            # Prefer closer to rich_qa (hi); use -abs(distance) so larger is better.
            prox = -abs(hi_ref - window_positions[wi])
            score = (ra_overlap, slug_overlap, ans_in, prox)
            if score > best_score:
                best_score = score
                best_wi = wi
        if best_wi != -1:
            window_claimed[best_wi] = True
            assignments[sid] = (best_wi, window_bodies[best_wi])
            top = best_score[0]
            top_ra_scores.append(top)
            if top >= 0.5:
                score_hist["ra_ge_0.5"] += 1
            elif top >= 0.3:
                score_hist["ra_ge_0.3"] += 1
            elif top >= 0.1:
                score_hist["ra_ge_0.1"] += 1
            else:
                score_hist["ra_lt_0.1"] += 1
        else:
            unresolved.append(sid)
    print(f"assigned by composite score: {len(assignments)}, unresolved in-range: {len(unresolved)}")
    print(f"top rich_answer overlap histogram: {score_hist}")

    # Fallback: for rows not anchored or with empty range, assign any unclaimed
    # window at the log position closest to their rich_qa_pos (if any).
    for sid in df["id"]:
        if sid in assignments:
            continue
        if sid in rich_qa_pos_by_id:
            # Search the nearest unclaimed window before rich_qa_pos globally.
            hi = rich_qa_pos_by_id[sid]
            idx = bisect.bisect_left(window_positions, hi) - 1
            while idx >= 0 and window_claimed[idx]:
                idx -= 1
            if idx >= 0:
                window_claimed[idx] = True
                assignments[sid] = (idx, window_bodies[idx])

    final_present = sum(1 for sid in df["id"] if sid in assignments)
    print(f"final windows present: {final_present}/{len(df)}")

    records = []
    for _, row in df.iterrows():
        sid = row["id"]
        info = assignments.get(sid)
        records.append(
            {
                "id": sid,
                "source_url": id_to_source_url[sid],
                "original_answer": id_to_orig_ans[sid],
                "seed": id_to_seed[sid],
                "window": info[1] if info else None,
                "window_found": info is not None,
            }
        )
    out = pd.DataFrame(records)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
