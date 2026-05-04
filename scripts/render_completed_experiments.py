#!/usr/bin/env python3
"""Render completed_experiments.tsv into a readable markdown file, latest first."""
import argparse
import csv
from pathlib import Path

DEFAULT_TSV = Path("/work/hdd/bbsg/twei2/rl/verl/logs/chain/completed_experiments.tsv")
DEFAULT_MD = Path("/work/hdd/bbsg/twei2/rl/verl/logs/chain/completed_experiments.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_MD)
    args = ap.parse_args()

    with args.tsv.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    rows.sort(key=lambda r: r.get("finished_at", ""), reverse=True)

    lines = [
        f"# Completed Experiments ({len(rows)})",
        "",
        f"Source: `{args.tsv}`  ",
        "Sorted latest first.",
        "",
        "| Finished | Run ID | Iters | Job ID | Script | Chain Dir |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        script = r.get("target_script", "")
        chain = r.get("chain_dir", "")
        lines.append(
            f"| {r.get('finished_at','')} "
            f"| {r.get('run_id','')} "
            f"| {r.get('iterations','')} "
            f"| {r.get('final_job_id','')} "
            f"| `{script}` "
            f"| `{chain}` |"
        )

    args.out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
