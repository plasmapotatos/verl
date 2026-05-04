"""Verify quality of rich_qa id -> window assignments.

Checks, in order of reliability:

1. rich_answer token overlap — the correct window was used as the passage
   when generating the rich_answer, so the rich_answer's content tokens
   should overlap heavily with the window.
2. original_answer substring — the original_answer must be in the window
   by construction (find_answer_window).
3. URL slug tokens — the URL's page title appears in the window when the
   answer is near the top of the article; otherwise the truncated window
   may not contain the slug even when the assignment is correct.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

WINDOWS = Path("data/simpleqa/partition/rich_qa/sft/train_windows.parquet")
PARQUET = Path("data/simpleqa/partition/rich_qa/sft/train.parquet")

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
TOK_RE = re.compile(r"[A-Za-z0-9]{3,}")


def tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOK_RE.finditer(text or "")}


def url_slug_tokens(url: str) -> list[str]:
    if "/wiki/" not in url:
        return []
    slug = url.split("/wiki/")[-1].split("#")[0].split("?")[0]
    try:
        slug = unquote(slug)
    except Exception:
        pass
    return [t.lower() for t in re.findall(r"[A-Za-z]+", slug.replace("_", " ")) if len(t) >= 3]


def main() -> None:
    df_w = pd.read_parquet(WINDOWS)
    df_p = pd.read_parquet(PARQUET)
    rich_a_by_id = {row["id"]: row["answer"] for _, row in df_p.iterrows()}

    print(f"rows: {len(df_w)}, windows present: {df_w['window_found'].sum()}")

    ra_overlaps: list[float] = []
    hist = {"ge_0.9": 0, "ge_0.7": 0, "ge_0.5": 0, "ge_0.3": 0, "lt_0.3": 0}
    ans_in = 0
    ans_tot = 0
    url_tok_all = 0
    url_tok_partial = 0
    url_tok_none = 0
    url_no_tok = 0
    suspicious: list[tuple[str, float]] = []

    for _, row in df_w.iterrows():
        if not row["window_found"] or not row["window"]:
            continue
        win = row["window"]
        win_low = win.lower()
        win_toks = tokenize(win_low)

        ra_toks = {t for t in tokenize(rich_a_by_id[row["id"]]) if t not in _STOP}
        if ra_toks:
            ov = len(ra_toks & win_toks) / len(ra_toks)
        else:
            ov = 0.0
        ra_overlaps.append(ov)
        if ov >= 0.9:
            hist["ge_0.9"] += 1
        elif ov >= 0.7:
            hist["ge_0.7"] += 1
        elif ov >= 0.5:
            hist["ge_0.5"] += 1
        elif ov >= 0.3:
            hist["ge_0.3"] += 1
        else:
            hist["lt_0.3"] += 1
            suspicious.append((row["id"], ov))

        orig_a = (row["original_answer"] or "").lower().strip()
        if orig_a:
            ans_tot += 1
            if orig_a in win_low:
                ans_in += 1

        stoks = url_slug_tokens(row["source_url"])
        if not stoks:
            url_no_tok += 1
        else:
            hits = sum(1 for t in stoks if t in win_low)
            if hits == len(stoks):
                url_tok_all += 1
            elif hits >= 1:
                url_tok_partial += 1
            else:
                url_tok_none += 1

    avg = sum(ra_overlaps) / len(ra_overlaps) if ra_overlaps else 0.0
    print(f"\nrich_answer token overlap: mean={avg:.3f}")
    print(f"  >=0.9: {hist['ge_0.9']}")
    print(f"  0.7-0.9: {hist['ge_0.7']}")
    print(f"  0.5-0.7: {hist['ge_0.5']}")
    print(f"  0.3-0.5: {hist['ge_0.3']}")
    print(f"  <0.3  : {hist['lt_0.3']}")
    print(f"\noriginal_answer in window: {ans_in}/{ans_tot}")
    print(f"\nURL slug tokens:")
    print(f"  all    : {url_tok_all}")
    print(f"  partial: {url_tok_partial}")
    print(f"  none   : {url_tok_none}")
    print(f"  no tok : {url_no_tok}")

    print(f"\nSuspicious (ra_overlap<0.3) samples: {len(suspicious)}")
    for sid, ov in suspicious[:10]:
        row = df_w[df_w["id"] == sid].iloc[0]
        print(f"  id={sid} ra_overlap={ov:.2f}")
        print(f"    url={row['source_url']}")
        print(f"    rich_a={rich_a_by_id[sid][:200]}")
        print(f"    window[:200]={row['window'][:200]}")


if __name__ == "__main__":
    main()
