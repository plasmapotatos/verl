#!/usr/bin/env python3
"""Build an interactive HTML explorer for pass@k trajectories across RL checkpoints.

Reads a directory of the form::

    <exp_dir>/global_step_<N>/pass@k/<split>/eval.json

and produces a single self-contained HTML page that shows:

  * Aggregate pass@k / accuracy vs step for every split (train-eval, val, ...).
  * A per-sample heatmap (rows = samples, cols = steps, color = #correct / k)
    with multiple sort orders (first-correct step, emergence slope, ...).
  * Per-sample drill-down: question, gold answer, every response at every step
    color-coded by grade (correct / incorrect / not_attempted / failed).
  * Transition stats: samples that went 0 -> correct, flipped back, etc.

Usage::

    python scripts/analyze_pass_at_k.py \\
        --exp-dir /work/hdd/bbsg/twei2/rl/verl/shared/unknown_decomp_grpo \\
        --out /work/hdd/bbsg/twei2/rl/verl/shared/unknown_decomp_grpo/pass_at_k_explorer.html
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


GRADE_CODE = {"correct": "C", "incorrect": "I", "not_attempted": "N", "failed_to_parse": "F", None: "?"}


def _grade_of(g):
    if not isinstance(g, dict):
        return "?"
    ev = g.get("evaluation")
    if isinstance(ev, str):
        return GRADE_CODE.get(ev.lower(), "?")
    return "?"


def _truncate(s, n=300):
    s = "" if s is None else str(s)
    s = s.replace("\r", "").strip()
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def discover_steps(exp_dir):
    steps = []
    for p in exp_dir.glob("global_step_*/pass@k"):
        m = re.match(r"global_step_(\d+)$", p.parent.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(set(steps))


def discover_splits(exp_dir, steps):
    splits = set()
    for s in steps:
        root = exp_dir / f"global_step_{s}" / "pass@k"
        if root.exists():
            for sub in root.iterdir():
                if sub.is_dir() and (sub / "eval.json").exists():
                    splits.add(sub.name)
    return sorted(splits)


def load_split(exp_dir, split, steps, resp_chars):
    samples = {}
    step_metrics = {}
    eval_data_paths = set()

    for s in steps:
        eval_path = exp_dir / f"global_step_{s}" / "pass@k" / split / "eval.json"
        pk_path = exp_dir / f"global_step_{s}" / "pass@k" / split / "pass_at_k.json"
        if not eval_path.exists():
            continue
        with open(eval_path) as f:
            ev = json.load(f)
        pk = {}
        if pk_path.exists():
            with open(pk_path) as f:
                pk = json.load(f)
                eval_data_paths.add(pk.get("eval_data", ""))

        step_metrics[s] = {
            "response_metrics": pk.get("response_metrics", {}),
            "pass_at_k": pk.get("pass_at_k", {}),
            "k": pk.get("k"),
        }

        for row in ev.get("rows", []):
            sid = str(row.get("id"))
            graders = row.get("graders") or []
            responses = row.get("coerced_responses") or row.get("responses") or []
            grades = [_grade_of(g) for g in graders]
            predicted = [str(g.get("predicted_answer", "")) if isinstance(g, dict) else "" for g in graders]
            correct = sum(1 for c in grades if c == "C")
            attempted = sum(1 for c in grades if c in ("C", "I"))
            k = len(grades)

            samp = samples.setdefault(
                sid,
                {
                    "id": sid,
                    "question": row.get("question") or (row.get("extra_info", {}) or {}).get("question") or "",
                    "answer": row.get("answer")
                    or (row.get("reward_model", {}) or {}).get("ground_truth")
                    or (row.get("extra_info", {}) or {}).get("answer")
                    or "",
                    "metadata": row.get("metadata") or (row.get("extra_info", {}) or {}).get("metadata") or {},
                    "per_step": {},
                },
            )
            samp["per_step"][str(s)] = {
                "k": k,
                "correct": correct,
                "attempted": attempted,
                "grades": "".join(grades),
                "predicted": [_truncate(p, 120) for p in predicted],
                "responses": [_truncate(r, resp_chars) for r in responses] if resp_chars > 0 else [],
            }

    return {
        "split": split,
        "eval_data_paths": sorted(p for p in eval_data_paths if p),
        "steps": sorted(step_metrics.keys()),
        "step_metrics": step_metrics,
        "samples": samples,
    }


def build_payload(exp_dir, resp_chars):
    steps = discover_steps(exp_dir)
    splits = discover_splits(exp_dir, steps)
    data = {}
    for split in splits:
        data[split] = load_split(exp_dir, split, steps, resp_chars=resp_chars)
    return {
        "exp_dir": str(exp_dir),
        "steps": steps,
        "splits": splits,
        "by_split": data,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>pass@k explorer &mdash; __EXP_NAME__</title>
<style>
  :root {
    --bg: #0f1115; --fg: #e6e6e6; --muted:#9aa3b2; --panel:#171a21;
    --c-correct:#22c55e; --c-incorrect:#ef4444; --c-not:#6b7280; --c-failed:#a855f7; --c-unknown:#374151;
    --accent:#38bdf8;
  }
  html,body { background:var(--bg); color:var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; margin:0; }
  header { padding:14px 20px; border-bottom:1px solid #222; display:flex; gap:20px; align-items:baseline; flex-wrap:wrap; }
  header h1 { font-size:16px; margin:0; font-weight:600; }
  header .meta { color:var(--muted); font-size:12px; font-family: ui-monospace, Menlo, monospace; }
  .controls { padding:10px 20px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; background:var(--panel); border-bottom:1px solid #222; }
  .controls label { font-size:12px; color:var(--muted); }
  .controls select, .controls input { background:#0b0d12; color:var(--fg); border:1px solid #2a2f3a; border-radius:4px; padding:4px 6px; font-size:12px; }
  main { padding:16px 20px; display:grid; grid-template-columns: minmax(480px, 1.1fr) minmax(420px, 1fr); gap:18px; }
  .card { background:var(--panel); border:1px solid #222; border-radius:6px; padding:12px; }
  .card h2 { font-size:13px; margin:0 0 10px 0; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; font-weight:600; }
  table.metrics { width:100%; border-collapse:collapse; font-size:12px; font-family: ui-monospace, Menlo, monospace; }
  table.metrics th, table.metrics td { padding:4px 6px; text-align:right; border-bottom:1px solid #222; }
  table.metrics th:first-child, table.metrics td:first-child { text-align:left; color:var(--muted); }
  .heatmap-wrap { overflow:auto; max-height:620px; border:1px solid #222; border-radius:4px; }
  svg { display:block; }
  .tooltip { position:fixed; pointer-events:none; background:#000; color:#fff; padding:6px 8px; font-size:11px;
             font-family: ui-monospace, Menlo, monospace; border:1px solid #333; border-radius:3px; max-width:420px; z-index:10; white-space:pre-wrap; }
  .legend { display:flex; gap:14px; font-size:11px; color:var(--muted); margin-top:6px; flex-wrap:wrap; }
  .legend span b { display:inline-block; width:12px; height:12px; vertical-align:-2px; margin-right:4px; border:1px solid #000; }
  .drill h3 { font-size:13px; margin:4px 0; }
  .drill .q { font-family: ui-monospace, Menlo, monospace; font-size:12px; color:#cbd5e1; background:#0b0d12; border:1px solid #222; border-radius:4px; padding:8px; white-space:pre-wrap; }
  .drill .ans { color:var(--c-correct); font-weight:600; }
  .step-block { margin-top:10px; }
  .step-block .hdr { font-size:11px; color:var(--muted); font-family: ui-monospace, Menlo, monospace; margin-bottom:3px; }
  .chips { display:flex; flex-wrap:wrap; gap:3px; }
  .chip { font-family: ui-monospace, Menlo, monospace; font-size:11px; padding:2px 6px; border-radius:3px; border:1px solid #000; max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; cursor:help; }
  .chip.C { background:#164e2b; color:#b8f0c8; border-color:#0f3c20; }
  .chip.I { background:#3a1414; color:#ff9a9a; border-color:#5a1f1f; }
  .chip.N { background:#1f2430; color:#9ca3af; }
  .chip.F { background:#2a1236; color:#d8b4fe; }
  .chip._unknown { background:#1f2937; color:#6b7280; }
  .tabs { display:flex; gap:4px; margin-bottom:10px; flex-wrap:wrap; }
  .tab { padding:4px 10px; font-size:12px; background:#0b0d12; border:1px solid #2a2f3a; border-radius:4px; cursor:pointer; color:var(--muted); line-height:1.2; text-align:center; min-width:40px; }
  .tab.active { background:var(--accent); color:#000; border-color:var(--accent); font-weight:600; }
  .stat-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(140px,1fr)); gap:8px; }
  .stat { background:#0b0d12; border:1px solid #222; border-radius:4px; padding:6px 8px; }
  .stat .v { font-family: ui-monospace, Menlo, monospace; font-size:18px; }
  .stat .l { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }
  .linechart { width:100%; height:220px; }
  .muted { color:var(--muted); }
  input[type=range] { vertical-align:middle; }
</style>
</head>
<body>
<header>
  <h1>pass@k trajectory explorer</h1>
  <div class="meta">__EXP_DIR__</div>
</header>

<div class="controls">
  <label>split:
    <select id="split"></select>
  </label>
  <label>sort:
    <select id="sort">
      <option value="first_correct">first-correct step (ascending)</option>
      <option value="final_correct">final #correct (descending)</option>
      <option value="improvement">improvement (final - initial)</option>
      <option value="volatility">volatility (std over steps)</option>
      <option value="id">sample id</option>
    </select>
  </label>
  <label>filter:
    <select id="filter">
      <option value="all">all samples</option>
      <option value="emerged">emerged (0 &rarr; &gt;0 correct)</option>
      <option value="regressed">regressed (peak &gt; final)</option>
      <option value="stuck">stuck at 0</option>
      <option value="mastered">mastered (final &ge; 0.9k)</option>
    </select>
  </label>
  <label>row px:
    <input id="rowpx" type="range" min="2" max="14" value="6" />
  </label>
  <span class="meta" id="counts"></span>
</div>

<main>
  <section>
    <div class="card">
      <h2>aggregate metrics vs step</h2>
      <div id="chart"></div>
      <div class="legend">
        <span><b style="background:#f59e0b"></b>train-like split</span>
        <span><b style="background:#38bdf8"></b>current split</span>
        <span class="muted">solid = pass@k, dashed = mean accuracy(of-k)</span>
      </div>
      <div class="stat-grid" id="statgrid" style="margin-top:12px"></div>
    </div>

    <div class="card" style="margin-top:16px">
      <h2>sample heatmap (click a cell to drill down)</h2>
      <div class="heatmap-wrap" id="heat"></div>
      <div class="legend">
        <span>intensity = #correct / k. white border marks first-correct step.</span>
      </div>
    </div>
  </section>

  <section>
    <div class="card">
      <h2>transition analysis</h2>
      <div id="transitions"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h2>drill-down</h2>
      <div class="tabs" id="drill-tabs"></div>
      <div class="drill" id="drill"></div>
    </div>
  </section>
</main>

<div id="tooltip" class="tooltip" style="display:none"></div>

<script>
const DATA = __PAYLOAD_JSON__;
const state = { split: DATA.splits[0], sort: "first_correct", filter: "all", rowpx: 6, selected: null, drillTab: null };

const CODE_TO_LABEL = {"C":"correct","I":"incorrect","N":"not_attempted","F":"failed_to_parse","?":"unknown"};

const tooltip = document.getElementById("tooltip");
function showTip(ev, txt) { tooltip.textContent = txt; tooltip.style.display="block"; tooltip.style.left=(ev.clientX+12)+"px"; tooltip.style.top=(ev.clientY+12)+"px"; }
function hideTip() { tooltip.style.display="none"; }

function escapeHtml(s) { return ("" + (s==null?"":s)).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }

function init() {
  const splitSel = document.getElementById("split");
  for (const s of DATA.splits) {
    const o = document.createElement("option"); o.value=s; o.textContent=s; splitSel.appendChild(o);
  }
  // default to a val-like split if present
  const valLike = DATA.splits.find(s => /held|val/.test(s));
  if (valLike) { state.split = valLike; splitSel.value = valLike; }
  splitSel.addEventListener("change", () => { state.split = splitSel.value; state.selected=null; render(); });
  document.getElementById("sort").addEventListener("change", e => { state.sort = e.target.value; render(); });
  document.getElementById("filter").addEventListener("change", e => { state.filter = e.target.value; render(); });
  document.getElementById("rowpx").addEventListener("input", e => { state.rowpx = +e.target.value; renderHeatmap(); });
  render();
}

function sampleArray() {
  const sp = DATA.by_split[state.split];
  const steps = sp.steps.map(String);
  return Object.values(sp.samples).map(s => {
    const seq = steps.map(st => {
      const ps = s.per_step[st];
      return ps ? ps.correct / (ps.k || 1) : null;
    });
    const raw = steps.map(st => (s.per_step[st] ? s.per_step[st].correct : null));
    const kArr = steps.map(st => s.per_step[st] ? s.per_step[st].k : 0);
    const k = kArr.find(v => v) || 32;
    const firstCorrectIdx = seq.findIndex(v => v !== null && v > 0);
    const valid = seq.filter(v => v !== null);
    const peak = valid.length ? Math.max(...valid) : 0;
    const initial = seq[0] ?? 0;
    const final = seq[seq.length-1] ?? 0;
    const mean = valid.reduce((a,b)=>a+b,0) / Math.max(1,valid.length);
    const std = Math.sqrt(valid.reduce((a,b)=>a+(b-mean)**2,0) / Math.max(1,valid.length));
    return { sample:s, seq, raw, k, firstCorrectIdx, peak, initial, final, std };
  });
}

function applyFilter(arr) {
  if (state.filter === "emerged") return arr.filter(r => r.initial === 0 && r.final > 0);
  if (state.filter === "regressed") return arr.filter(r => r.peak > r.final + 1e-9);
  if (state.filter === "stuck") return arr.filter(r => r.peak === 0);
  if (state.filter === "mastered") return arr.filter(r => r.final >= 0.9);
  return arr;
}

function applySort(arr) {
  const s = state.sort;
  const key = fn => arr.sort((a,b) => fn(a)-fn(b));
  if (s === "first_correct") return key(r => r.firstCorrectIdx === -1 ? 9999 : r.firstCorrectIdx + r.final*0.001);
  if (s === "final_correct") return key(r => -r.final);
  if (s === "improvement") return key(r => -(r.final - r.initial));
  if (s === "volatility") return key(r => -r.std);
  if (s === "id") return arr.sort((a,b) => (""+a.sample.id).localeCompare(""+b.sample.id, undefined, {numeric:true}));
  return arr;
}

function intensityColor(v) {
  if (v === null || v === undefined) return "#111";
  if (v === 0) return "#2a1010";
  const g = Math.round(40 + v*180);
  const r = Math.round(180 * (1 - v));
  return `rgb(${r},${g},60)`;
}

function render() {
  renderChart();
  renderStats();
  renderHeatmap();
  renderTransitions();
  renderDrill();
}

function renderChart() {
  const W = 600, H = 220, M = {t:10, r:10, b:30, l:40};
  const iw = W-M.l-M.r, ih = H-M.t-M.b;
  const allSteps = DATA.steps;
  const xMax = Math.max(...allSteps, 1);
  const x = v => M.l + (v/xMax)*iw;
  const y = v => M.t + (1-v)*ih;

  function ser(splitKey, field) {
    const s = DATA.by_split[splitKey];
    if (!s) return [];
    return s.steps.map(st => {
      const m = s.step_metrics[st];
      if (!m) return null;
      const v = field === "pass"
        ? (m.pass_at_k && m.pass_at_k.pass_at_k)
        : (m.response_metrics && m.response_metrics.accuracy);
      return v == null ? null : { x: +st, y: v };
    }).filter(Boolean);
  }
  const trainLike = DATA.splits.find(s => s !== state.split && /train/.test(s));
  const plots = [];
  if (trainLike) plots.push({key: trainLike, color:"#f59e0b"});
  plots.push({key: state.split, color:"#38bdf8"});

  let svg = `<svg class="linechart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">`;
  svg += `<g><line x1="${M.l}" y1="${M.t+ih}" x2="${M.l+iw}" y2="${M.t+ih}" stroke="#333"/>`;
  for (let t=0; t<=1.001; t+=0.25) svg += `<line x1="${M.l}" y1="${y(t)}" x2="${M.l+iw}" y2="${y(t)}" stroke="#222"/><text x="${M.l-4}" y="${y(t)+3}" text-anchor="end" fill="#9aa3b2" font-size="10" font-family="ui-monospace">${t.toFixed(2)}</text>`;
  for (const st of allSteps) svg += `<text x="${x(st)}" y="${M.t+ih+14}" text-anchor="middle" fill="#9aa3b2" font-size="10" font-family="ui-monospace">${st}</text>`;
  svg += `</g>`;

  for (const p of plots) {
    for (const [field, dash] of [["pass",""],["acc","4 3"]]) {
      const pts = ser(p.key, field);
      if (!pts.length) continue;
      const d = pts.map((pp,i) => `${i?"L":"M"}${x(pp.x).toFixed(1)},${y(pp.y).toFixed(1)}`).join("");
      svg += `<path d="${d}" fill="none" stroke="${p.color}" stroke-width="1.8" stroke-dasharray="${dash}"/>`;
      for (const pp of pts) svg += `<circle cx="${x(pp.x)}" cy="${y(pp.y)}" r="2.5" fill="${p.color}"><title>${p.key} ${field}@${pp.x}=${pp.y.toFixed(3)}</title></circle>`;
    }
  }
  svg += `</svg>`;
  document.getElementById("chart").innerHTML = svg;
}

function renderStats() {
  const sp = DATA.by_split[state.split];
  const steps = sp.steps;
  const first = steps[0], last = steps[steps.length-1];
  const pk = step => (sp.step_metrics[step] && sp.step_metrics[step].pass_at_k && sp.step_metrics[step].pass_at_k.pass_at_k) || 0;
  const ac = step => (sp.step_metrics[step] && sp.step_metrics[step].response_metrics && sp.step_metrics[step].response_metrics.accuracy) || 0;
  const ar = step => (sp.step_metrics[step] && sp.step_metrics[step].response_metrics && sp.step_metrics[step].response_metrics.attempt_rate) || 0;
  const N = Object.keys(sp.samples).length;

  const arr = sampleArray();
  const emerged = arr.filter(r => r.initial===0 && r.final>0).length;
  const regressed = arr.filter(r => r.peak > r.final + 1e-9).length;
  const stuck = arr.filter(r => r.peak === 0).length;
  const mastered = arr.filter(r => r.final >= 0.9).length;

  const rows = [
    ["samples", N],
    ["pass@k "+first, pk(first).toFixed(3)],
    ["pass@k "+last, pk(last).toFixed(3)],
    ["\u0394 pass@k", ((pk(last)-pk(first))>=0?"+":"") + (pk(last)-pk(first)).toFixed(3)],
    ["accuracy "+last, ac(last).toFixed(3)],
    ["attempt-rate "+last, ar(last).toFixed(3)],
    ["emerged", emerged],
    ["regressed", regressed],
    ["stuck-at-0", stuck],
    ["mastered", mastered],
  ];
  document.getElementById("statgrid").innerHTML = rows.map(([l,v]) =>
    `<div class="stat"><div class="v">${v}</div><div class="l">${l}</div></div>`
  ).join("");
  document.getElementById("counts").textContent = `n=${N}  split=${state.split}  eval=${(sp.eval_data_paths||[]).join(", ")}`;
}

function renderHeatmap() {
  const sp = DATA.by_split[state.split];
  const steps = sp.steps;
  const colW = 28, rowH = state.rowpx, leftPad = 60, topPad = 24;
  let arr = applySort(applyFilter(sampleArray()));
  const W = leftPad + steps.length * colW + 10;
  const H = topPad + arr.length * rowH + 10;
  let svg = `<svg width="${W}" height="${H}">`;
  for (let i=0;i<steps.length;i++) {
    svg += `<text x="${leftPad + i*colW + colW/2}" y="${topPad-8}" text-anchor="middle" fill="#9aa3b2" font-size="10" font-family="ui-monospace">${steps[i]}</text>`;
  }
  for (let r=0; r<arr.length; r++) {
    const row = arr[r];
    const y = topPad + r*rowH;
    if (row.sample.id === state.selected) {
      svg += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="#1e3a5f"/>`;
    }
    if (rowH >= 9) svg += `<text x="${leftPad-4}" y="${y+rowH-1}" text-anchor="end" fill="#6b7280" font-size="${Math.min(rowH-1,10)}" font-family="ui-monospace">${row.sample.id}</text>`;
    for (let i=0;i<steps.length;i++) {
      const v = row.seq[i];
      const raw = row.raw[i];
      const fill = intensityColor(v);
      const xx = leftPad + i*colW;
      const border = (i === row.firstCorrectIdx) ? "#fff" : "none";
      svg += `<rect class="cell" data-id="${row.sample.id}" data-step="${steps[i]}" x="${xx}" y="${y}" width="${colW-1}" height="${Math.max(rowH-0.5,1)}" fill="${fill}" stroke="${border}" stroke-width="${border==="none"?0:1}"><title>id=${row.sample.id}  step=${steps[i]}  ${raw==null?"-":raw}/${row.k}</title></rect>`;
    }
  }
  svg += `</svg>`;
  const el = document.getElementById("heat");
  el.innerHTML = svg;
  el.querySelectorAll("rect.cell").forEach(rect => {
    rect.addEventListener("click", () => {
      state.selected = rect.getAttribute("data-id");
      state.drillTab = rect.getAttribute("data-step");
      renderHeatmap(); renderDrill();
    });
  });
}

function renderTransitions() {
  const sp = DATA.by_split[state.split];
  const steps = sp.steps;
  const arr = sampleArray();
  const rows = [];
  rows.push(`<table class="metrics"><thead><tr><th>step pair</th><th>\u0394 pass@k</th><th>newly solved</th><th>newly lost</th><th>mean \u0394 correct/k</th></tr></thead><tbody>`);
  for (let i=0;i<steps.length-1;i++) {
    const a = String(steps[i]), b = String(steps[i+1]);
    let gained=0, lost=0, dsum=0, n=0;
    for (const r of arr) {
      const pa = r.sample.per_step[a], pb = r.sample.per_step[b];
      if (!pa || !pb) continue;
      const ra = pa.correct/(pa.k||1), rb = pb.correct/(pb.k||1);
      if (ra === 0 && rb > 0) gained++;
      if (ra > 0 && rb === 0) lost++;
      dsum += (rb-ra); n++;
    }
    const pkA = ((sp.step_metrics[steps[i]]||{}).pass_at_k||{}).pass_at_k || 0;
    const pkB = ((sp.step_metrics[steps[i+1]]||{}).pass_at_k||{}).pass_at_k || 0;
    rows.push(`<tr><td>${a} \u2192 ${b}</td><td>${(pkB-pkA>=0?"+":"")}${(pkB-pkA).toFixed(3)}</td><td>${gained}</td><td>${lost}</td><td>${(dsum/Math.max(1,n)).toFixed(3)}</td></tr>`);
  }
  rows.push(`</tbody></table>`);

  const buckets = Array(steps.length+1).fill(0);
  for (const r of arr) buckets[r.firstCorrectIdx === -1 ? steps.length : r.firstCorrectIdx]++;
  const maxB = Math.max(...buckets, 1);
  const bw = 46, bh = 80, pad=6;
  let hist = `<h3 style="font-size:12px; color:var(--muted); margin:12px 0 4px 0;">first-correct step (last bar = never solved)</h3><svg width="${(steps.length+1)*bw+pad*2}" height="${bh+30}">`;
  for (let i=0;i<=steps.length;i++) {
    const h = (buckets[i]/maxB)*bh;
    const xx = pad + i*bw, yy = bh-h+4;
    hist += `<rect x="${xx}" y="${yy}" width="${bw-6}" height="${h}" fill="${i===steps.length?"#555":"#38bdf8"}"/>`;
    hist += `<text x="${xx+(bw-6)/2}" y="${yy-2}" text-anchor="middle" font-size="10" fill="#9aa3b2" font-family="ui-monospace">${buckets[i]}</text>`;
    hist += `<text x="${xx+(bw-6)/2}" y="${bh+20}" text-anchor="middle" font-size="10" fill="#9aa3b2" font-family="ui-monospace">${i===steps.length?"never":steps[i]}</text>`;
  }
  hist += `</svg>`;

  // Stacked area: count of samples with fraction-correct in bins [0], (0,.25], (.25,.5], (.5,.75], (.75,1)
  const bins = 5;
  const binEdges = [0, 0.0001, 0.25, 0.5, 0.75, 0.9999, 1.01];
  const binColors = ["#2a1010","#602020","#b55233","#c9a834","#7bc96f","#22c55e"];
  const binLabels = ["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "=1"];
  const countsByStep = steps.map(st => {
    const c = new Array(binEdges.length-1).fill(0);
    for (const r of arr) {
      const ps = r.sample.per_step[String(st)];
      if (!ps) continue;
      const v = ps.correct/(ps.k||1);
      let idx = 0;
      if (v === 0) idx = 0;
      else if (v >= 1) idx = 5;
      else if (v <= 0.25) idx = 1;
      else if (v <= 0.5) idx = 2;
      else if (v <= 0.75) idx = 3;
      else idx = 4;
      c[idx]++;
    }
    return c;
  });
  const N = arr.length || 1;
  const sw = 520, sh = 120, spad = 30;
  const iw = sw - spad*2, ih = sh - 20;
  let stack = `<h3 style="font-size:12px; color:var(--muted); margin:12px 0 4px 0;">distribution of per-sample #correct/k over training</h3><svg width="${sw}" height="${sh+20}">`;
  for (let i=0;i<steps.length;i++) {
    const x1 = spad + (i/(steps.length-1 || 1))*iw;
    let y = ih;
    for (let b=0;b<binEdges.length-1;b++) {
      const frac = countsByStep[i][b]/N;
      const h = frac*ih;
      y -= h;
      const nextX = i<steps.length-1 ? spad + ((i+1)/(steps.length-1 || 1))*iw : x1+2;
      stack += `<rect x="${x1}" y="${y}" width="${nextX-x1}" height="${h}" fill="${binColors[b]}"><title>step ${steps[i]} bin ${binLabels[b]}: ${countsByStep[i][b]}</title></rect>`;
    }
    stack += `<text x="${x1}" y="${ih+14}" text-anchor="middle" font-size="10" fill="#9aa3b2" font-family="ui-monospace">${steps[i]}</text>`;
  }
  // legend
  for (let b=0;b<binLabels.length;b++) {
    stack += `<rect x="${spad + b*72}" y="${sh+2}" width="10" height="10" fill="${binColors[b]}"/><text x="${spad + b*72 + 14}" y="${sh+12}" font-size="10" fill="#9aa3b2" font-family="ui-monospace">${binLabels[b]}</text>`;
  }
  stack += `</svg>`;

  document.getElementById("transitions").innerHTML = rows.join("") + hist + stack;
}

function renderDrill() {
  const drill = document.getElementById("drill");
  const tabs = document.getElementById("drill-tabs");
  if (!state.selected) { drill.innerHTML = `<div class="muted">click a heatmap cell to inspect a sample.</div>`; tabs.innerHTML=""; return; }
  const sp = DATA.by_split[state.split];
  const s = sp.samples[state.selected];
  if (!s) { drill.innerHTML = `<div class="muted">sample ${state.selected} not found.</div>`; return; }
  const steps = sp.steps;
  if (!state.drillTab || !s.per_step[state.drillTab]) state.drillTab = String(steps[0]);

  tabs.innerHTML = steps.map(st => {
    const ps = s.per_step[String(st)];
    const lbl = ps ? `${st}<br><span style="font-size:10px">${ps.correct}/${ps.k}</span>` : String(st);
    return `<div class="tab ${String(st)===String(state.drillTab)?"active":""}" data-step="${st}">${lbl}</div>`;
  }).join("");
  tabs.querySelectorAll(".tab").forEach(t => t.addEventListener("click", () => { state.drillTab = t.getAttribute("data-step"); renderDrill(); }));

  const ps = s.per_step[String(state.drillTab)];
  const meta = s.metadata || {};
  let h = "";
  h += `<h3>sample ${s.id}</h3>`;
  h += `<div class="q"><b>Q:</b> ${escapeHtml(s.question)}\n<b>A:</b> <span class="ans">${escapeHtml(s.answer)}</span>`;
  if (meta && (meta.topic || meta.answer_type)) h += `\n<b>meta:</b> ${escapeHtml("topic="+(meta.topic||"?")+"  answer_type="+(meta.answer_type||"?"))}`;
  h += `</div>`;

  // sparkline
  const trajW = 420, trajH = 28;
  let traj = `<svg width="${trajW}" height="${trajH+16}">`;
  const cw = trajW / Math.max(1,steps.length);
  for (let i=0;i<steps.length;i++) {
    const p = s.per_step[String(steps[i])];
    const v = p ? p.correct/(p.k||1) : null;
    traj += `<rect x="${i*cw}" y="2" width="${cw-1}" height="${trajH-4}" fill="${intensityColor(v)}" stroke="${String(steps[i])===String(state.drillTab)?"#fff":"none"}"/>`;
    traj += `<text x="${i*cw + cw/2}" y="${trajH+10}" text-anchor="middle" font-size="9" fill="#9aa3b2" font-family="ui-monospace">${steps[i]}</text>`;
  }
  traj += `</svg>`;
  h += `<div class="step-block"><div class="hdr">trajectory</div>${traj}</div>`;

  if (ps) {
    h += `<div class="step-block"><div class="hdr">step ${state.drillTab}: ${ps.correct}/${ps.k} correct, ${ps.attempted}/${ps.k} attempted</div><div class="chips">`;
    for (let i=0;i<ps.grades.length;i++) {
      const g = ps.grades[i];
      const cls = g === "?" ? "_unknown" : g;
      const pred = (ps.predicted && ps.predicted[i]) || "";
      const resp = (ps.responses && ps.responses[i]) || "";
      const tip = `grade=${CODE_TO_LABEL[g]||g}\npred=${pred}\n---\n${resp}`;
      h += `<span class="chip ${cls}" data-tip="${escapeHtml(tip)}">${escapeHtml(pred || "(empty)")}</span>`;
    }
    h += `</div></div>`;
  }

  drill.innerHTML = h;
  drill.querySelectorAll(".chip[data-tip]").forEach(c => {
    c.addEventListener("mousemove", ev => showTip(ev, c.getAttribute("data-tip")));
    c.addEventListener("mouseleave", hideTip);
  });
}

init();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resp-chars", type=int, default=240, help="chars of raw response to embed (0 = skip)")
    args = ap.parse_args()

    payload = build_payload(args.exp_dir, resp_chars=args.resp_chars)
    if not payload["steps"]:
        raise SystemExit(f"no global_step_*/pass@k directories found under {args.exp_dir}")

    html_doc = HTML_TEMPLATE.replace("__EXP_NAME__", html.escape(args.exp_dir.name))
    html_doc = html_doc.replace("__EXP_DIR__", html.escape(str(args.exp_dir)))
    html_doc = html_doc.replace("__PAYLOAD_JSON__", json.dumps(payload))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_doc)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.1f} MB)")
    print(f"  splits: {payload['splits']}")
    print(f"  steps:  {payload['steps']}")
    for split, d in payload["by_split"].items():
        n = len(d["samples"])
        print(f"  {split}: {n} samples, eval_data={d['eval_data_paths']}")


if __name__ == "__main__":
    main()
