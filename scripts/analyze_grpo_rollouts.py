#!/usr/bin/env python3
"""Build an interactive explorer for GRPO rollout trajectories."""

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path

try:
    import datasets  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    datasets = None

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    pd = None

try:
    import pyarrow.parquet as pq  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    pq = None


def _to_jsonable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _normalize_text(text):
    text = "" if text is None else str(text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_question_from_prompt_text(text):
    if not text:
        return ""

    matches = re.findall(r"(?:^|\n)user\n(.*?)(?:\nassistant(?:\n|$))", text, flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip()

    marker = "\nassistant"
    if marker in text:
        before_assistant = text.split(marker, 1)[0]
        if "\nuser\n" in before_assistant:
            return before_assistant.rsplit("\nuser\n", 1)[-1].strip()

    return text.strip()


def _extract_question_from_dataset_row(row):
    question = row.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()

    prompt = row.get("prompt")
    if isinstance(prompt, list):
        user_messages = []
        for item in prompt:
            if not isinstance(item, dict):
                continue
            if item.get("role") == "user":
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    user_messages.append(content.strip())
        if user_messages:
            return user_messages[-1]

    if isinstance(prompt, str) and prompt.strip():
        return _extract_question_from_prompt_text(prompt)

    return ""


def _extract_answer_from_dataset_row(row):
    answer = row.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        if isinstance(ground_truth, str) and ground_truth.strip():
            return ground_truth.strip()

    return ""


def _iter_dataset_rows(dataset_path):
    dataset_path = str(dataset_path)

    if datasets is not None:
        dataset = datasets.load_dataset("parquet", data_files=dataset_path)["train"]
        for index in range(len(dataset)):
            yield index, _to_jsonable(dataset[index])
        return

    if pd is not None:
        dataframe = pd.read_parquet(dataset_path)
        for index, (_, row) in enumerate(dataframe.iterrows()):
            yield index, _to_jsonable(row.to_dict())
        return

    if pq is not None:
        table = pq.read_table(dataset_path)
        for index, row in enumerate(table.to_pylist()):
            yield index, _to_jsonable(row)
        return

    raise RuntimeError(
        "No parquet reader is available. Install one of: datasets, pandas, or pyarrow."
    )


def _make_sample_key(sample_id, sample_index):
    if sample_id is not None and str(sample_id) != "":
        return "id:%s" % sample_id
    return "index:%s" % sample_index


def _rollout_file_sort_key(path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.name)
    match = re.search(r"(\d+)", stem)
    if match:
        return (1, int(match.group(1)), path.name)
    return (2, stem, path.name)


def _mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def _format_score(score):
    if score is None:
        return "n/a"
    return "{:.3f}".format(score)


def _build_dataset_index(dataset_path):
    sample_index = []
    sample_by_key = {}
    question_to_keys = defaultdict(list)
    id_to_key = {}
    index_to_key = {}

    for sample_index_num, row in _iter_dataset_rows(dataset_path):
        sample_id = row.get("id")
        question = _extract_question_from_dataset_row(row)
        answer = _extract_answer_from_dataset_row(row)
        normalized_question = _normalize_text(question)
        sample_key = _make_sample_key(sample_id, sample_index_num)

        meta = {
            "sample_key": sample_key,
            "sample_id": None if sample_id is None else str(sample_id),
            "sample_index": sample_index_num,
            "question": question,
            "answer": answer,
            "normalized_question": normalized_question,
            "search_blob": _normalize_text(
                "%s %s %s %s" % (sample_id if sample_id is not None else "", sample_index_num, question, answer)
            ),
            "duplicate_question_count": 0,
            "has_rollouts": False,
            "match_count": 0,
            "step_count": 0,
            "first_step": None,
            "last_step": None,
            "best_score": None,
            "mean_score": None,
        }
        sample_index.append(meta)
        sample_by_key[sample_key] = meta
        index_to_key[str(sample_index_num)] = sample_key
        if sample_id is not None:
            id_to_key[str(sample_id)] = sample_key
        if normalized_question:
            question_to_keys[normalized_question].append(sample_key)

    for keys in question_to_keys.values():
        dup_count = len(keys)
        for sample_key in keys:
            sample_by_key[sample_key]["duplicate_question_count"] = dup_count

    return {
        "sample_index": sample_index,
        "sample_by_key": sample_by_key,
        "question_to_keys": question_to_keys,
        "id_to_key": id_to_key,
        "index_to_key": index_to_key,
    }


def _scan_rollouts(rollout_dir, question_to_keys):
    rollout_dir = Path(rollout_dir)
    rollout_files = sorted(rollout_dir.glob("*.jsonl"), key=_rollout_file_sort_key)
    matches_by_key = defaultdict(list)
    prompt_by_key = {}
    unmatched_rollouts = 0
    unmatched_questions = defaultdict(int)
    total_rollouts = 0

    for file_path in rollout_files:
        fallback_step = None
        if file_path.stem.isdigit():
            fallback_step = int(file_path.stem)

        with file_path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_rollouts += 1
                prompt_text = payload.get("input", "") if isinstance(payload, dict) else ""
                question = _extract_question_from_prompt_text(prompt_text)
                normalized_question = _normalize_text(question)
                sample_keys = question_to_keys.get(normalized_question, [])
                if not sample_keys:
                    unmatched_rollouts += 1
                    if normalized_question:
                        unmatched_questions[normalized_question] += 1
                    continue

                sample_key = sample_keys[0]
                if sample_key not in prompt_by_key and prompt_text:
                    prompt_by_key[sample_key] = prompt_text

                step = payload.get("step", fallback_step)
                try:
                    step = int(step)
                except Exception:
                    step = fallback_step if fallback_step is not None else -1

                score = payload.get("score")
                try:
                    score = float(score)
                except Exception:
                    score = None

                matches_by_key[sample_key].append(
                    {
                        "step": step,
                        "score": score,
                        "output": payload.get("output", ""),
                        "rollout_file": file_path.name,
                        "line_number": line_no,
                    }
                )

    for sample_key in matches_by_key:
        matches_by_key[sample_key].sort(
            key=lambda item: (item.get("step", -1), item.get("rollout_file", ""), item.get("line_number", 0))
        )

    unmatched_examples = []
    for normalized_question, count in sorted(unmatched_questions.items(), key=lambda item: (-item[1], item[0]))[:10]:
        unmatched_examples.append({"normalized_question": normalized_question, "count": count})

    return {
        "matches_by_key": matches_by_key,
        "prompt_by_key": prompt_by_key,
        "rollout_file_count": len(rollout_files),
        "total_rollouts": total_rollouts,
        "unmatched_rollouts": unmatched_rollouts,
        "unmatched_examples": unmatched_examples,
    }


def _summarize_steps(matches):
    grouped = defaultdict(list)
    for match in matches:
        grouped[match["step"]].append(match)

    step_rows = []
    for step in sorted(grouped.keys()):
        items = grouped[step]
        scores = [item["score"] for item in items if item["score"] is not None]
        outputs = [item.get("output", "") for item in items]
        best_item = None
        if items:
            best_item = max(
                items,
                key=lambda item: (
                    float("-inf") if item["score"] is None else item["score"],
                    -len(item.get("output", "")),
                ),
            )
        step_rows.append(
            {
                "step": step,
                "count": len(items),
                "mean_score": _mean(scores),
                "max_score": max(scores) if scores else None,
                "min_score": min(scores) if scores else None,
                "positive_count": sum(1 for score in scores if score > 0),
                "zero_count": sum(1 for score in scores if score == 0),
                "unique_outputs": len(set(outputs)),
                "best_output": best_item.get("output", "") if best_item else "",
            }
        )

    return step_rows


def _sample_overall(matches, step_rows):
    scores = [item["score"] for item in matches if item["score"] is not None]
    steps = [row["step"] for row in step_rows]
    return {
        "match_count": len(matches),
        "step_count": len(step_rows),
        "first_step": min(steps) if steps else None,
        "last_step": max(steps) if steps else None,
        "best_score": max(scores) if scores else None,
        "mean_score": _mean(scores),
    }


def _resolve_initial_sample_key(args, dataset_index, sample_by_key, matches_by_key):
    if args.sample_id is not None:
        sample_key = dataset_index["id_to_key"].get(str(args.sample_id))
        if sample_key is None:
            raise KeyError("Dataset row not found for sample-id=%s" % args.sample_id)
        return sample_key

    if args.sample_index is not None:
        sample_key = dataset_index["index_to_key"].get(str(args.sample_index))
        if sample_key is None:
            raise KeyError("Dataset row not found for sample-index=%s" % args.sample_index)
        return sample_key

    if args.question is not None:
        normalized_question = _normalize_text(args.question)
        sample_keys = dataset_index["question_to_keys"].get(normalized_question, [])
        if not sample_keys:
            raise KeyError("Dataset row not found for question=%s" % args.question)
        return sample_keys[0]

    matched_entries = [sample_by_key[key] for key in matches_by_key.keys() if key in sample_by_key]
    if matched_entries:
        matched_entries.sort(
            key=lambda item: (
                -(item.get("match_count") or 0),
                -(item.get("best_score") if item.get("best_score") is not None else -1),
                item.get("sample_index", 0),
            )
        )
        return matched_entries[0]["sample_key"]

    if sample_by_key:
        return sorted(sample_by_key.values(), key=lambda item: item.get("sample_index", 0))[0]["sample_key"]

    return None


def _default_output_base(rollout_dir):
    rollout_name = Path(rollout_dir).resolve().name
    return Path("outputs") / "rollout_analysis" / rollout_name / "explorer"


def _build_explorer(args):
    dataset_path = Path(args.dataset).expanduser().resolve()
    rollout_dir = Path(args.rollout_dir).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError("Dataset not found: %s" % dataset_path)
    if not rollout_dir.is_dir():
        raise FileNotFoundError("Rollout directory not found: %s" % rollout_dir)

    dataset_index = _build_dataset_index(dataset_path)
    sample_index = dataset_index["sample_index"]
    sample_by_key = dataset_index["sample_by_key"]

    rollout_scan = _scan_rollouts(rollout_dir, dataset_index["question_to_keys"])
    matches_by_key = rollout_scan["matches_by_key"]
    prompt_by_key = rollout_scan["prompt_by_key"]

    sample_data = {}
    matched_sample_count = 0

    for sample_key, matches in matches_by_key.items():
        meta = sample_by_key.get(sample_key)
        if meta is None:
            continue

        step_rows = _summarize_steps(matches)
        overall = _sample_overall(matches, step_rows)
        meta["has_rollouts"] = bool(matches)
        meta["match_count"] = overall["match_count"]
        meta["step_count"] = overall["step_count"]
        meta["first_step"] = overall["first_step"]
        meta["last_step"] = overall["last_step"]
        meta["best_score"] = overall["best_score"]
        meta["mean_score"] = overall["mean_score"]
        matched_sample_count += 1

        sample_data[sample_key] = {
            "target": {
                "sample_key": sample_key,
                "sample_id": meta["sample_id"],
                "sample_index": meta["sample_index"],
                "question": meta["question"],
                "answer": meta["answer"],
                "normalized_question": meta["normalized_question"],
                "prompt_text": prompt_by_key.get(sample_key, ""),
                "duplicate_question_count": meta["duplicate_question_count"],
            },
            "overall": overall,
            "by_step": step_rows,
            "matches": matches,
        }

    for meta in sample_index:
        if not meta["has_rollouts"]:
            meta["match_count"] = 0
            meta["step_count"] = 0
            meta["first_step"] = None
            meta["last_step"] = None
            meta["best_score"] = None
            meta["mean_score"] = None

    initial_sample_key = _resolve_initial_sample_key(args, dataset_index, sample_by_key, matches_by_key)

    return {
        "version": 1,
        "dataset_path": str(dataset_path),
        "rollout_dir": str(rollout_dir),
        "max_rollouts_per_step_preview": max(1, args.max_rollouts_per_step),
        "build_stats": {
            "dataset_row_count": len(sample_index),
            "rollout_file_count": rollout_scan["rollout_file_count"],
            "total_rollouts": rollout_scan["total_rollouts"],
            "matched_sample_count": matched_sample_count,
            "matched_rollout_count": rollout_scan["total_rollouts"] - rollout_scan["unmatched_rollouts"],
            "unmatched_rollout_count": rollout_scan["unmatched_rollouts"],
            "unmatched_examples": rollout_scan["unmatched_examples"],
        },
        "initial_sample_key": initial_sample_key,
        "sample_index": sample_index,
        "sample_data": sample_data,
    }


def _render_html(explorer, html_output):
    explorer_json = json.dumps(explorer, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape("GRPO Rollout Explorer")

    html_template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #efe7d5;
      --panel: #fffdf7;
      --line: #d7c7ab;
      --ink: #221d17;
      --muted: #6f685d;
      --blue: #1b6ca8;
      --green: #2d7c31;
      --gold: #a76f17;
      --rose: #8f3b3b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at top left, rgba(255, 247, 224, 0.96), transparent 34%),
        linear-gradient(180deg, #efe5d0, #f7f1e6 30%, #efe8db 100%);
    }
    .wrap {
      max-width: 1450px;
      margin: 0 auto;
      padding: 18px;
    }
    .hero,
    .panel,
    .result-item {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 14px 36px rgba(70, 52, 29, 0.08);
    }
    .hero {
      padding: 20px 22px;
      margin-bottom: 16px;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 34px;
      line-height: 1.08;
    }
    h2 {
      margin: 0 0 12px 0;
      font-size: 22px;
    }
    h3 {
      margin: 0;
      font-size: 18px;
    }
    .sub {
      color: var(--muted);
      line-height: 1.5;
    }
    .build-meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-top: 14px;
    }
    .meta-card {
      padding: 10px 12px;
      border: 1px solid #eadfc9;
      border-radius: 12px;
      background: #fcfaf4;
    }
    .meta-label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .meta-value {
      margin-top: 4px;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .controls {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 16px;
    }
    input, select, button {
      font: inherit;
      border-radius: 10px;
    }
    input, select {
      min-height: 40px;
      padding: 8px 10px;
      border: 1px solid #d8ccb8;
      background: #fffdf9;
      color: var(--ink);
    }
    button {
      min-height: 40px;
      padding: 8px 14px;
      border: 1px solid #c7b289;
      background: linear-gradient(180deg, #f3dfb5, #e7c682);
      color: #3f2d10;
      cursor: pointer;
    }
    button.secondary {
      background: linear-gradient(180deg, #f5f4f0, #e5e2da);
      border-color: #cec7bb;
      color: #463e34;
    }
    .layout {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }
    .sidebar,
    .main {
      display: grid;
      gap: 16px;
    }
    .panel {
      padding: 18px 20px;
    }
    .result-list {
      display: grid;
      gap: 10px;
      max-height: 80vh;
      overflow: auto;
      padding-right: 2px;
    }
    .result-item {
      padding: 12px 14px;
      cursor: pointer;
      transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
    }
    .result-item:hover {
      transform: translateY(-1px);
      border-color: #b89453;
    }
    .result-item.active {
      border-color: #a76f17;
      box-shadow: 0 10px 26px rgba(146, 104, 30, 0.16);
    }
    .result-top {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      font-size: 13px;
      color: var(--muted);
    }
    .result-question {
      margin-top: 6px;
      font-size: 15px;
      line-height: 1.45;
    }
    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 12px;
      border: 1px solid #e7dac1;
      background: #faf5e8;
      color: #5f5547;
    }
    .pill.good {
      background: #edf7ef;
      border-color: #c6e0c9;
      color: #2d6d31;
    }
    .pill.warn {
      background: #f8efe6;
      border-color: #e8cfb3;
      color: #8a5a19;
    }
    .sample-title {
      font-size: 26px;
      line-height: 1.25;
      margin: 0 0 10px 0;
      white-space: pre-wrap;
    }
    .sample-answer {
      color: #423b33;
      white-space: pre-wrap;
      margin-bottom: 14px;
    }
    .trend-wrap {
      margin-top: 10px;
      border: 1px solid #ebe0cc;
      border-radius: 14px;
      overflow: hidden;
      background: #fffdf8;
    }
    .step-grid {
      display: grid;
      gap: 14px;
    }
    .step-card {
      border: 1px solid #ebdfca;
      border-radius: 14px;
      background: #fcfaf5;
      padding: 14px;
    }
    .step-header {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      align-items: baseline;
    }
    .step-stats {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }
    details {
      margin-top: 10px;
      border-top: 1px solid #eadfcf;
      padding-top: 10px;
    }
    summary {
      cursor: pointer;
      color: var(--gold);
      font-weight: 600;
    }
    pre {
      margin: 8px 0 0 0;
      padding: 12px;
      border-radius: 12px;
      border: 1px solid #eee2cf;
      background: #fbf8f2;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 12.5px;
      line-height: 1.5;
    }
    .raw-controls {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
      margin-bottom: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      text-align: left;
      vertical-align: top;
      padding: 10px 8px;
      border-bottom: 1px solid #efe4d2;
      font-size: 13px;
    }
    th {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .muted { color: var(--muted); }
    .empty {
      padding: 14px;
      border-radius: 12px;
      background: #faf7ef;
      border: 1px dashed #dfcfb3;
      color: var(--muted);
    }
    .footer-note {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      .result-list { max-height: none; }
    }
    @media (max-width: 700px) {
      .wrap { padding: 12px; }
      h1 { font-size: 28px; }
      .sample-title { font-size: 21px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>__TITLE__</h1>
      <div class="sub">
        One build, then search by sample ID, dataset row index, or question text directly in the page.
      </div>
      <div class="build-meta" id="buildMeta"></div>
      <div class="controls">
        <select id="lookupMode">
          <option value="any">Search anything</option>
          <option value="id">Sample ID</option>
          <option value="index">Sample index</option>
          <option value="question">Question text</option>
        </select>
        <input id="lookupInput" type="text" placeholder="Enter sample ID, index, or question text" style="flex:1;min-width:220px" />
        <button id="openBtn">Open Best Match</button>
        <button id="prevBtn" class="secondary">Prev Matched</button>
        <button id="nextBtn" class="secondary">Next Matched</button>
      </div>
      <div class="footer-note" id="statusLine"></div>
    </section>

    <div class="layout">
      <aside class="sidebar">
        <section class="panel">
          <h2>Search Results</h2>
          <div class="sub">Top matches update as you type.</div>
          <div class="result-list" id="resultList"></div>
        </section>
      </aside>

      <main class="main">
        <section class="panel">
          <div id="sampleHero"></div>
        </section>

        <section class="panel">
          <h2>Reward Trend</h2>
          <div class="sub">Blue is mean reward by step. Green is max reward by step.</div>
          <div class="trend-wrap" id="trendWrap"></div>
        </section>

        <section class="panel">
          <h2>Prompt</h2>
          <div id="promptBlock"></div>
        </section>

        <section class="panel">
          <h2>Step Summaries</h2>
          <div class="step-grid" id="stepGrid"></div>
        </section>

        <section class="panel">
          <h2>Raw Rollouts</h2>
          <div class="raw-controls">
            <label for="outputSearch">Filter output:</label>
            <input id="outputSearch" type="text" placeholder="Search outputs for the current sample" style="flex:1;min-width:220px" />
            <label for="stepFilter">Step:</label>
            <select id="stepFilter"></select>
            <label for="scoreFilter">Min score:</label>
            <select id="scoreFilter">
              <option value="">Any</option>
              <option value="0">0</option>
              <option value="0.5">0.5</option>
              <option value="1">1</option>
            </select>
          </div>
          <table>
            <thead>
              <tr>
                <th>Step</th>
                <th>Score</th>
                <th>Source</th>
                <th>Output</th>
              </tr>
            </thead>
            <tbody id="rawTableBody"></tbody>
          </table>
        </section>
      </main>
    </div>
  </div>

  <script>
    const explorer = __EXPLORER_JSON__;
    const sampleIndex = explorer.sample_index || [];
    const sampleData = explorer.sample_data || {};
    const maxPreview = explorer.max_rollouts_per_step_preview || 12;
    const sampleByKey = {};
    sampleIndex.forEach(entry => {
      sampleByKey[entry.sample_key] = entry;
    });

    const matchedKeys = sampleIndex
      .filter(entry => entry.has_rollouts)
      .sort((a, b) => {
        const matchDiff = (b.match_count || 0) - (a.match_count || 0);
        if (matchDiff) return matchDiff;
        return (a.sample_index || 0) - (b.sample_index || 0);
      })
      .map(entry => entry.sample_key);

    let currentKey = explorer.initial_sample_key || (matchedKeys[0] || (sampleIndex[0] && sampleIndex[0].sample_key) || null);
    let currentResults = [];

    const buildMeta = document.getElementById('buildMeta');
    const resultList = document.getElementById('resultList');
    const lookupMode = document.getElementById('lookupMode');
    const lookupInput = document.getElementById('lookupInput');
    const openBtn = document.getElementById('openBtn');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const statusLine = document.getElementById('statusLine');
    const sampleHero = document.getElementById('sampleHero');
    const trendWrap = document.getElementById('trendWrap');
    const promptBlock = document.getElementById('promptBlock');
    const stepGrid = document.getElementById('stepGrid');
    const outputSearch = document.getElementById('outputSearch');
    const stepFilter = document.getElementById('stepFilter');
    const scoreFilter = document.getElementById('scoreFilter');
    const rawTableBody = document.getElementById('rawTableBody');

    function normalizeText(text) {
      return String(text || '')
        .trim()
        .toLowerCase()
        .replace(/\s+/g, ' ')
        .replace(/[^a-z0-9]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
    }

    function escapeHtml(value) {
      return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    function formatScore(score) {
      if (score == null || Number.isNaN(score)) return 'n/a';
      return Number(score).toFixed(3);
    }

    function buildMetaCards() {
      const stats = explorer.build_stats || {};
      const cards = [
        ['Dataset rows', stats.dataset_row_count],
        ['Rollout files', stats.rollout_file_count],
        ['Total rollouts', stats.total_rollouts],
        ['Matched samples', stats.matched_sample_count],
        ['Matched rollouts', stats.matched_rollout_count],
        ['Unmatched rollouts', stats.unmatched_rollout_count],
      ];
      buildMeta.innerHTML = cards.map(([label, value]) => (
        `<div class="meta-card"><div class="meta-label">${escapeHtml(label)}</div><div class="meta-value">${escapeHtml(value)}</div></div>`
      )).join('');
    }

    function getDetail(sampleKey) {
      const meta = sampleByKey[sampleKey];
      const detail = sampleData[sampleKey];
      if (detail) return detail;
      return {
        target: {
          sample_key: sampleKey,
          sample_id: meta ? meta.sample_id : null,
          sample_index: meta ? meta.sample_index : null,
          question: meta ? meta.question : '',
          answer: meta ? meta.answer : '',
          normalized_question: meta ? meta.normalized_question : '',
          prompt_text: '',
          duplicate_question_count: meta ? meta.duplicate_question_count : 0,
        },
        overall: {
          match_count: 0,
          step_count: 0,
          first_step: null,
          last_step: null,
          best_score: null,
          mean_score: null,
        },
        by_step: [],
        matches: [],
      };
    }

    function searchEntries(query, mode) {
      const normalizedQuery = normalizeText(query);
      let rows = sampleIndex.slice();
      if (!normalizedQuery) {
        rows = rows.filter(entry => entry.has_rollouts);
        rows.sort((a, b) => {
          const matchDiff = (b.match_count || 0) - (a.match_count || 0);
          if (matchDiff) return matchDiff;
          return (a.sample_index || 0) - (b.sample_index || 0);
        });
        return rows.slice(0, 120);
      }

      const scored = [];
      rows.forEach(entry => {
        const sampleId = String(entry.sample_id == null ? '' : entry.sample_id);
        const sampleIndexText = String(entry.sample_index == null ? '' : entry.sample_index);
        const questionNorm = entry.normalized_question || '';
        const answerNorm = normalizeText(entry.answer || '');
        let score = 0;

        if (mode === 'id' || mode === 'any') {
          if (sampleId === query.trim()) score = Math.max(score, 100);
          else if (sampleId && sampleId.indexOf(query.trim()) >= 0) score = Math.max(score, 70);
        }
        if (mode === 'index' || mode === 'any') {
          if (sampleIndexText === query.trim()) score = Math.max(score, 95);
          else if (sampleIndexText.indexOf(query.trim()) >= 0) score = Math.max(score, 65);
        }
        if (mode === 'question' || mode === 'any') {
          if (questionNorm === normalizedQuery) score = Math.max(score, 98);
          else if (questionNorm.indexOf(normalizedQuery) >= 0) score = Math.max(score, 75);
          else if (normalizedQuery.indexOf(questionNorm) >= 0 && questionNorm) score = Math.max(score, 50);
          else if (answerNorm && answerNorm.indexOf(normalizedQuery) >= 0 && mode === 'any') score = Math.max(score, 40);
          else if ((entry.search_blob || '').indexOf(normalizedQuery) >= 0 && mode === 'any') score = Math.max(score, 30);
        }

        if (score > 0) {
          scored.push({ entry, score });
        }
      });

      scored.sort((a, b) => {
        const scoreDiff = b.score - a.score;
        if (scoreDiff) return scoreDiff;
        const rolloutDiff = (b.entry.match_count || 0) - (a.entry.match_count || 0);
        if (rolloutDiff) return rolloutDiff;
        return (a.entry.sample_index || 0) - (b.entry.sample_index || 0);
      });

      return scored.slice(0, 120).map(item => item.entry);
    }

    function renderResults() {
      currentResults = searchEntries(lookupInput.value, lookupMode.value);
      if (!currentResults.length) {
        resultList.innerHTML = '<div class="empty">No matching dataset rows were found.</div>';
        statusLine.textContent = 'No search results.';
        return;
      }

      statusLine.textContent = `${currentResults.length} search result${currentResults.length === 1 ? '' : 's'} shown.`;
      resultList.innerHTML = currentResults.map(entry => {
        const active = entry.sample_key === currentKey ? ' active' : '';
        const answer = entry.answer ? `<div class="muted">${escapeHtml(entry.answer)}</div>` : '';
        const pills = [
          `<span class="pill">${escapeHtml(entry.sample_id == null ? 'id: n/a' : 'id: ' + entry.sample_id)}</span>`,
          `<span class="pill">row: ${escapeHtml(entry.sample_index)}</span>`,
          `<span class="pill ${entry.has_rollouts ? 'good' : 'warn'}">${entry.has_rollouts ? escapeHtml((entry.match_count || 0) + ' rollouts') : 'no rollouts'}</span>`,
          `<span class="pill">steps: ${escapeHtml(entry.step_count || 0)}</span>`
        ].join('');
        return `
          <div class="result-item${active}" data-sample-key="${escapeHtml(entry.sample_key)}">
            <div class="result-top">
              <span>${escapeHtml(entry.sample_key)}</span>
              <span>${escapeHtml(formatScore(entry.best_score))}</span>
            </div>
            <div class="result-question">${escapeHtml(entry.question || '(no question)')}</div>
            ${answer}
            <div class="pill-row">${pills}</div>
          </div>
        `;
      }).join('');

      Array.from(resultList.querySelectorAll('[data-sample-key]')).forEach(node => {
        node.addEventListener('click', () => {
          chooseSample(node.getAttribute('data-sample-key'), true);
        });
      });
    }

    function matchedKeyIndex(sampleKey) {
      return matchedKeys.indexOf(sampleKey);
    }

    function chooseNeighbor(direction) {
      if (!matchedKeys.length) return;
      const currentIndex = matchedKeyIndex(currentKey);
      if (currentIndex < 0) {
        chooseSample(matchedKeys[0], true);
        return;
      }
      const nextIndex = Math.max(0, Math.min(matchedKeys.length - 1, currentIndex + direction));
      chooseSample(matchedKeys[nextIndex], true);
    }

    function chooseBestMatch() {
      const query = lookupInput.value.trim();
      const mode = lookupMode.value;
      if (!query) {
        if (currentResults.length) {
          chooseSample(currentResults[0].sample_key, true);
        }
        return;
      }

      if (mode === 'id') {
        const direct = sampleIndex.find(entry => String(entry.sample_id || '') === query);
        if (direct) {
          chooseSample(direct.sample_key, true);
          return;
        }
      }
      if (mode === 'index') {
        const direct = sampleIndex.find(entry => String(entry.sample_index) === query);
        if (direct) {
          chooseSample(direct.sample_key, true);
          return;
        }
      }
      if (mode === 'question') {
        const direct = sampleIndex.find(entry => entry.normalized_question === normalizeText(query));
        if (direct) {
          chooseSample(direct.sample_key, true);
          return;
        }
      }

      const results = searchEntries(query, mode);
      if (results.length) {
        chooseSample(results[0].sample_key, true);
      }
    }

    function buildTrendSvg(stepRows) {
      if (!stepRows.length) {
        return '<div class="empty">No rollout matches were found for this sample.</div>';
      }

      const width = 900;
      const height = 250;
      const left = 54;
      const right = 18;
      const top = 18;
      const bottom = 34;
      const innerWidth = width - left - right;
      const innerHeight = height - top - bottom;
      const xs = stepRows.map(row => row.step);
      const ys = [];
      stepRows.forEach(row => {
        if (row.mean_score != null) ys.push(Number(row.mean_score));
        if (row.max_score != null) ys.push(Number(row.max_score));
      });
      if (!ys.length) ys.push(0);

      const minY = Math.min(0, ...ys);
      const maxY = Math.max(1, ...ys);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const xSpan = Math.max(1, maxX - minX);
      const ySpan = Math.max(1, maxY - minY);

      function point(step, score) {
        const x = left + ((step - minX) / xSpan) * innerWidth;
        const y = top + (1 - ((score - minY) / ySpan)) * innerHeight;
        return [x, y];
      }

      function polyline(key) {
        return stepRows
          .filter(row => row[key] != null)
          .map(row => {
            const [x, y] = point(row.step, Number(row[key]));
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          })
          .join(' ');
      }

      const grid = [0, 0.25, 0.5, 0.75, 1].map(frac => {
        const yValue = minY + (1 - frac) * ySpan;
        const [, y] = point(minX, yValue);
        return `
          <line x1="${left}" y1="${y.toFixed(1)}" x2="${width - right}" y2="${y.toFixed(1)}" stroke="#e7e7e1" stroke-width="1" />
          <text x="8" y="${(y + 4).toFixed(1)}" fill="#666" font-size="11">${escapeHtml(formatScore(yValue))}</text>
        `;
      }).join('');

      const ticks = stepRows.map(row => {
        const [x] = point(row.step, minY);
        return `<text x="${x.toFixed(1)}" y="${height - 10}" text-anchor="middle" fill="#666" font-size="11">${escapeHtml(row.step)}</text>`;
      }).join('');

      const circles = stepRows.map(row => {
        const nodes = [];
        if (row.mean_score != null) {
          const [x, y] = point(row.step, Number(row.mean_score));
          nodes.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.6" fill="#1b6ca8"></circle>`);
        }
        if (row.max_score != null) {
          const [x, y] = point(row.step, Number(row.max_score));
          nodes.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.1" fill="#2d7c31"></circle>`);
        }
        return nodes.join('');
      }).join('');

      return `
        <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img" aria-label="Reward trend by step">
          <rect x="0" y="0" width="${width}" height="${height}" fill="#fffdf8"></rect>
          ${grid}
          <line x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}" stroke="#9aa0a6" stroke-width="1.2"></line>
          <line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" stroke="#9aa0a6" stroke-width="1.2"></line>
          <polyline fill="none" stroke="#1b6ca8" stroke-width="2.6" points="${polyline('mean_score')}"></polyline>
          <polyline fill="none" stroke="#2d7c31" stroke-width="2.6" stroke-dasharray="6 4" points="${polyline('max_score')}"></polyline>
          ${ticks}
          ${circles}
        </svg>
      `;
    }

    function renderSampleHero(meta, detail) {
      const overall = detail.overall || {};
      const topCards = [
        ['Sample key', meta.sample_key],
        ['Sample id', meta.sample_id == null ? 'n/a' : meta.sample_id],
        ['Dataset row', meta.sample_index],
        ['Rollouts', overall.match_count || 0],
        ['Steps', overall.step_count || 0],
        ['First step', overall.first_step == null ? 'n/a' : overall.first_step],
        ['Last step', overall.last_step == null ? 'n/a' : overall.last_step],
        ['Best reward', formatScore(overall.best_score)],
        ['Mean reward', formatScore(overall.mean_score)],
        ['Duplicate question count', meta.duplicate_question_count || 0]
      ];

      sampleHero.innerHTML = `
        <div class="sample-title">${escapeHtml(meta.question || '(no question)')}</div>
        <div class="sample-answer"><strong>Ground truth:</strong> ${escapeHtml(meta.answer || 'n/a')}</div>
        <div class="build-meta">
          ${topCards.map(([label, value]) => `
            <div class="meta-card">
              <div class="meta-label">${escapeHtml(label)}</div>
              <div class="meta-value">${escapeHtml(value)}</div>
            </div>
          `).join('')}
        </div>
      `;
    }

    function renderPrompt(detail) {
      const promptText = detail.target && detail.target.prompt_text ? detail.target.prompt_text : detail.target.question;
      promptBlock.innerHTML = `<pre>${escapeHtml(promptText || '(no prompt captured)')}</pre>`;
    }

    function renderSteps(detail) {
      const byStep = detail.by_step || [];
      const matches = detail.matches || [];
      if (!byStep.length) {
        stepGrid.innerHTML = '<div class="empty">No rollout matches were found for this sample.</div>';
        return;
      }

      const matchesByStep = {};
      matches.forEach(match => {
        const key = String(match.step);
        if (!matchesByStep[key]) matchesByStep[key] = [];
        matchesByStep[key].push(match);
      });

      stepGrid.innerHTML = byStep.map(row => {
        const stepMatches = matchesByStep[String(row.step)] || [];
        const preview = stepMatches.slice(0, maxPreview);
        const remaining = Math.max(0, stepMatches.length - preview.length);
        return `
          <section class="step-card">
            <div class="step-header">
              <h3>Step ${escapeHtml(row.step)}</h3>
              <div class="step-stats">
                <span>count: ${escapeHtml(row.count)}</span>
                <span>mean: ${escapeHtml(formatScore(row.mean_score))}</span>
                <span>max: ${escapeHtml(formatScore(row.max_score))}</span>
                <span>unique outputs: ${escapeHtml(row.unique_outputs)}</span>
              </div>
            </div>
            <details open>
              <summary>Best output</summary>
              <pre>${escapeHtml(row.best_output || '')}</pre>
            </details>
            <details>
              <summary>Preview rollouts (${preview.length}${remaining ? ' of ' + stepMatches.length + ' shown' : ''})</summary>
              ${preview.length ? preview.map(match => `
                <div style="margin-top:10px;padding-top:10px;border-top:1px dashed #eadfcf">
                  <div class="muted">score=${escapeHtml(formatScore(match.score))} | ${escapeHtml(match.rollout_file)}:${escapeHtml(match.line_number)}</div>
                  <pre>${escapeHtml(match.output || '')}</pre>
                </div>
              `).join('') : '<div class="empty">No rollout previews available.</div>'}
            </details>
          </section>
        `;
      }).join('');
    }

    function fillStepFilter(detail) {
      const rows = detail.by_step || [];
      stepFilter.innerHTML = '<option value="">All</option>' + rows.map(row => (
        `<option value="${escapeHtml(row.step)}">${escapeHtml(row.step)}</option>`
      )).join('');
    }

    function renderRawRows(detail) {
      const query = normalizeText(outputSearch.value);
      const targetStep = stepFilter.value;
      const minScore = scoreFilter.value === '' ? null : Number(scoreFilter.value);
      const rows = (detail.matches || []).filter(row => {
        const text = normalizeText(row.output || '');
        if (query && text.indexOf(query) < 0) return false;
        if (targetStep !== '' && String(row.step) !== targetStep) return false;
        if (minScore != null && (row.score == null || Number(row.score) < minScore)) return false;
        return true;
      });

      if (!rows.length) {
        rawTableBody.innerHTML = '<tr><td colspan="4"><div class="empty">No rollout rows match the current filters.</div></td></tr>';
        return;
      }

      rawTableBody.innerHTML = rows.map(row => `
        <tr>
          <td>${escapeHtml(row.step)}</td>
          <td>${escapeHtml(formatScore(row.score))}</td>
          <td>${escapeHtml(row.rollout_file)}:${escapeHtml(row.line_number)}</td>
          <td><pre>${escapeHtml(row.output || '')}</pre></td>
        </tr>
      `).join('');
    }

    function renderCurrent() {
      if (!currentKey || !sampleByKey[currentKey]) {
        sampleHero.innerHTML = '<div class="empty">No sample selected.</div>';
        trendWrap.innerHTML = '<div class="empty">No sample selected.</div>';
        promptBlock.innerHTML = '<div class="empty">No sample selected.</div>';
        stepGrid.innerHTML = '<div class="empty">No sample selected.</div>';
        rawTableBody.innerHTML = '<tr><td colspan="4"><div class="empty">No sample selected.</div></td></tr>';
        return;
      }

      const meta = sampleByKey[currentKey];
      const detail = getDetail(currentKey);
      renderSampleHero(meta, detail);
      trendWrap.innerHTML = buildTrendSvg(detail.by_step || []);
      renderPrompt(detail);
      renderSteps(detail);
      fillStepFilter(detail);
      renderRawRows(detail);
      renderResults();

      const rolloutText = meta.has_rollouts
        ? `${meta.match_count || 0} rollout rows across ${meta.step_count || 0} steps`
        : 'dataset row is present, but no rollout rows matched it';
      statusLine.textContent = `${meta.sample_key} selected: ${rolloutText}.`;
    }

    function chooseSample(sampleKey, updateHash) {
      if (!sampleByKey[sampleKey]) return;
      currentKey = sampleKey;
      if (updateHash) {
        location.hash = 'sample=' + encodeURIComponent(sampleKey);
      }
      renderCurrent();
    }

    function syncFromHash() {
      const hash = location.hash || '';
      const prefix = '#sample=';
      if (!hash.startsWith(prefix)) return false;
      const key = decodeURIComponent(hash.slice(prefix.length));
      if (sampleByKey[key]) {
        currentKey = key;
        return true;
      }
      return false;
    }

    lookupInput.addEventListener('input', renderResults);
    lookupMode.addEventListener('change', renderResults);
    openBtn.addEventListener('click', chooseBestMatch);
    prevBtn.addEventListener('click', () => chooseNeighbor(-1));
    nextBtn.addEventListener('click', () => chooseNeighbor(1));
    outputSearch.addEventListener('input', () => renderRawRows(getDetail(currentKey)));
    stepFilter.addEventListener('change', () => renderRawRows(getDetail(currentKey)));
    scoreFilter.addEventListener('change', () => renderRawRows(getDetail(currentKey)));
    lookupInput.addEventListener('keydown', event => {
      if (event.key === 'Enter') {
        chooseBestMatch();
      }
    });
    window.addEventListener('hashchange', () => {
      if (syncFromHash()) renderCurrent();
    });

    buildMetaCards();
    renderResults();
    if (syncFromHash()) {
      renderCurrent();
    } else {
      renderCurrent();
      if (currentKey) {
        location.hash = 'sample=' + encodeURIComponent(currentKey);
      }
    }
  </script>
</body>
</html>
"""

    html_content = html_template.replace("__TITLE__", title).replace("__EXPLORER_JSON__", explorer_json)
    html_output.parent.mkdir(parents=True, exist_ok=True)
    with html_output.open("w", encoding="utf-8") as handle:
        handle.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Build an interactive GRPO rollout explorer.")
    parser.add_argument("--rollout-dir", required=True, help="Directory containing step JSONL rollout files.")
    parser.add_argument("--dataset", required=True, help="Dataset parquet used to source sample IDs/questions.")
    initial = parser.add_mutually_exclusive_group(required=False)
    initial.add_argument("--sample-id", default=None, help="Optional sample ID to open initially in the explorer.")
    initial.add_argument("--sample-index", type=int, default=None, help="Optional dataset row index to open initially.")
    initial.add_argument("--question", default=None, help="Optional question text to open initially.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path for the explorer data bundle.",
    )
    parser.add_argument(
        "--html-output",
        default=None,
        help="Output HTML path for the explorer UI.",
    )
    parser.add_argument(
        "--max-rollouts-per-step",
        type=int,
        default=12,
        help="How many rollouts to preview inside each step card.",
    )
    args = parser.parse_args()

    output_base = _default_output_base(args.rollout_dir)
    output_path = Path(args.output) if args.output else output_base.with_suffix(".json")
    html_output = Path(args.html_output) if args.html_output else output_base.with_suffix(".html")

    explorer = _build_explorer(args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(explorer, handle, ensure_ascii=False, separators=(",", ":"))

    _render_html(explorer, html_output)

    print("Wrote:", output_path)
    print("Wrote:", html_output)


if __name__ == "__main__":
    main()
