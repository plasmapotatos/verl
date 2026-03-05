#!/usr/bin/env python3
"""Compare two judge run JSONs and extract disagreements."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_run(path: str) -> Dict[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    by_id: Dict[str, dict] = {}
    for row in rows:
        coerced_id = str(row.get("coerced_id", ""))
        if not coerced_id:
            continue
        by_id[coerced_id] = row
    return by_id


def _extract_fields(row: dict) -> Tuple[str, str, List[str], List[str]]:
    question = row.get("question", "") or row.get("prompt", "")
    answer = row.get("answer", "")
    responses = row.get("coerced_responses", [])
    evaluations = []
    graders = row.get("graders", [])
    for grader in graders:
        evaluations.append(grader.get("evaluation", ""))
    return question, answer, responses, evaluations


def _default_output_path(run_a: str, run_b: str) -> Path:
    base_a = Path(run_a)
    base_b = Path(run_b)
    name = f"{base_a.stem}__vs__{base_b.stem}_disagreements.json"
    return base_a.with_name(name)


def _default_html_path(json_path: Path) -> Path:
    return json_path.with_suffix(".html")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-a", required=True, help="Path to first judge JSON")
    parser.add_argument("--run-b", required=True, help="Path to second judge JSON")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path for disagreements (defaults next to run-a)",
    )
    parser.add_argument(
        "--html-output",
        default=None,
        help="Output HTML path (defaults next to JSON output)",
    )
    args = parser.parse_args()

    run_a = _load_run(args.run_a)
    run_b = _load_run(args.run_b)

    disagreements = []
    common_ids = set(run_a.keys()) & set(run_b.keys())
    for sample_id in sorted(common_ids):
        row_a = run_a[sample_id]
        row_b = run_b[sample_id]

        question, answer, responses, evals_a = _extract_fields(row_a)
        _, _, _, evals_b = _extract_fields(row_b)

        if evals_a != evals_b:
            disagreements.append(
                {
                    "id": sample_id,
                    "question": question,
                    "answer": answer,
                    "responses": responses,
                    "evaluations": {
                        "run_a": evals_a,
                        "run_b": evals_b,
                    },
                }
            )

    output_path = Path(args.output) if args.output else _default_output_path(args.run_a, args.run_b)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"count": len(disagreements), "rows": disagreements}, handle, ensure_ascii=False, indent=2)

    print(f"Wrote: {output_path} (rows={len(disagreements)})")

    html_path = Path(args.html_output) if args.html_output else _default_html_path(output_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    rows_json = json.dumps(disagreements, ensure_ascii=False).replace("</", "<\\/")
    run_a_label = html.escape(args.run_a)
    run_b_label = html.escape(args.run_b)

    html_content = f"""
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>Judge Disagreements</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 16px; }}
        .controls {{ display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }}
        button {{ padding: 6px 10px; }}
        input {{ padding: 6px; }}
        .meta {{ color: #555; margin-bottom: 8px; }}
        .card {{ border: 1px solid #ddd; padding: 12px; border-radius: 6px; }}
        .label {{ font-weight: bold; margin-top: 8px; }}
        .responses {{ white-space: pre-wrap; }}
        .evals {{ white-space: pre-wrap; }}
    </style>
</head>
<body>
    <div class=\"meta\">
        <div>Run A: {run_a_label}</div>
        <div>Run B: {run_b_label}</div>
    </div>
    <div class=\"controls\">
        <button id=\"prevBtn\">Prev</button>
        <button id=\"nextBtn\">Next</button>
        <span id=\"counter\"></span>
        <label for=\"idInput\">Go to id:</label>
        <input id=\"idInput\" type=\"text\" placeholder=\"id\" />
        <button id=\"goBtn\">Go</button>
    </div>
    <div class=\"card\">
        <div class=\"label\">ID</div>
        <div id=\"rowId\"></div>
        <div class=\"label\">Question</div>
        <div id=\"rowQuestion\"></div>
        <div class=\"label\">Answer</div>
        <div id=\"rowAnswer\"></div>
        <div class=\"label\">Responses</div>
        <div id=\"rowResponses\" class=\"responses\"></div>
        <div class=\"label\">Evaluations (Run A)</div>
        <div id=\"rowEvalA\" class=\"evals\"></div>
        <div class=\"label\">Evaluations (Run B)</div>
        <div id=\"rowEvalB\" class=\"evals\"></div>
    </div>

    <script>
        const rows = {rows_json};
        let idx = 0;

        const counter = document.getElementById('counter');
        const rowId = document.getElementById('rowId');
        const rowQuestion = document.getElementById('rowQuestion');
        const rowAnswer = document.getElementById('rowAnswer');
        const rowResponses = document.getElementById('rowResponses');
        const rowEvalA = document.getElementById('rowEvalA');
        const rowEvalB = document.getElementById('rowEvalB');
        const idInput = document.getElementById('idInput');

        function render() {{
            if (!rows.length) {{
                counter.textContent = 'No disagreements.';
                return;
            }}
            const row = rows[idx];
            counter.textContent = ` ${'{'}idx + 1{'}'} / ${'{'}rows.length{'}'}`;
            rowId.textContent = row.id;
            rowQuestion.textContent = row.question || '';
            rowAnswer.textContent = row.answer || '';
            rowResponses.textContent = (row.responses || []).join('\\n---\\n');
            rowEvalA.textContent = (row.evaluations?.run_a || []).join('\\n');
            rowEvalB.textContent = (row.evaluations?.run_b || []).join('\\n');
        }}

        function goToIndex(newIdx) {{
            if (!rows.length) return;
            idx = Math.max(0, Math.min(rows.length - 1, newIdx));
            render();
        }}

        document.getElementById('prevBtn').addEventListener('click', () => goToIndex(idx - 1));
        document.getElementById('nextBtn').addEventListener('click', () => goToIndex(idx + 1));
        document.getElementById('goBtn').addEventListener('click', () => {{
            const target = idInput.value.trim();
            if (!target) return;
            const found = rows.findIndex(r => String(r.id) === target);
            if (found >= 0) {{
                goToIndex(found);
            }} else {{
                alert('ID not found');
            }}
        }});

        render();
    </script>
</body>
</html>
"""

    with html_path.open("w", encoding="utf-8") as handle:
        handle.write(html_content)

    print(f"Wrote: {html_path}")


if __name__ == "__main__":
    main()
