#!/usr/bin/env python3
"""Interactive HTML explorer for an RL experiment directory.

Scans::

    <exp_dir>/global_step_<N>/generations/*__on_<split>_eval.json   (eval tab)
    <exp_dir>/global_step_<N>/pass@k/<split>/eval.json              (pass@k tab)
    <exp_dir>/global_step_<N>/pass@k/<split>/pass_at_k.json

and, if an SFT train parquet is given, joins by `id` so each eval sample can
display its matching SFT training rows (multiple rows allowed per id).

Produces a single self-contained HTML page with two tabs (Eval, Pass@k), a
per-split selector, a sample-id search input, an accuracy-vs-step line chart
and a per-step table of grades/responses.

Usage::

    python scripts/analyze_rl_experiment.py \\
        --exp-dir outputs/rl/simpleqa_decompose_and_richqa_grpo/binary_decompose_and_richqa \\
        --sft-train data/simpleqa/partition/decompose_and_richqa/sft/train.parquet \\
        --out <exp_dir>/explorer.html
"""
from __future__ import annotations

import argparse
import html as _html
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


def _truncate(s, n):
    if n <= 0:
        return ""
    s = "" if s is None else str(s)
    s = s.replace("\r", "").strip()
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def _to_jsonable(obj):
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
    if np is not None and isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if np is not None and isinstance(obj, (np.integer,)):
        return int(obj)
    if np is not None and isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def discover_steps(exp_dir: Path):
    steps = []
    for p in exp_dir.glob("global_step_*"):
        m = re.match(r"global_step_(\d+)$", p.name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(set(steps))


EVAL_NAME_RE = re.compile(r"__on_(?P<split>.+)_eval\.json$")


def load_eval(exp_dir: Path, steps, resp_chars: int):
    """Returns {split: {steps: [...], step_metrics: {step: {...}}, samples: {id: {...}}}}."""
    splits: dict[str, dict] = {}
    for s in steps:
        gdir = exp_dir / f"global_step_{s}" / "generations"
        if not gdir.exists():
            continue
        for fp in sorted(gdir.glob("*_eval.json")):
            m = EVAL_NAME_RE.search(fp.name)
            if not m:
                continue
            split = m.group("split")
            try:
                with open(fp) as f:
                    d = json.load(f)
            except Exception:
                continue
            entry = splits.setdefault(split, {"step_metrics": {}, "samples": {}})
            entry["step_metrics"][s] = d.get("metrics", {})
            for row in d.get("rows", []):
                sid = str(row.get("id"))
                graders = row.get("graders") or []
                responses = row.get("coerced_responses") or row.get("responses") or []
                grades = [_grade_of(g) for g in graders]
                predicted = [str(g.get("predicted_answer", "")) if isinstance(g, dict) else "" for g in graders]
                correct = sum(1 for c in grades if c == "C")
                k = len(grades) or 1
                samp = entry["samples"].setdefault(
                    sid,
                    {
                        "id": sid,
                        "question": row.get("question") or (row.get("extra_info") or {}).get("question") or "",
                        "answer": row.get("answer")
                        or (row.get("reward_model") or {}).get("ground_truth")
                        or (row.get("extra_info") or {}).get("answer")
                        or "",
                        "metadata": _to_jsonable(row.get("metadata") or (row.get("extra_info") or {}).get("metadata") or {}),
                        "per_step": {},
                    },
                )
                samp["per_step"][str(s)] = {
                    "k": k,
                    "correct": correct,
                    "acc": correct / k,
                    "grades": "".join(grades),
                    "predicted": [_truncate(p, 200) for p in predicted],
                    "responses": [_truncate(r, resp_chars) for r in responses],
                }
    for split, entry in splits.items():
        entry["steps"] = sorted(entry["step_metrics"].keys())
    return splits


def load_pass_at_k(exp_dir: Path, steps, resp_chars: int):
    splits: dict[str, dict] = {}
    for s in steps:
        root = exp_dir / f"global_step_{s}" / "pass@k"
        if not root.exists():
            continue
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            ep = sub / "eval.json"
            pp = sub / "pass_at_k.json"
            if not ep.exists():
                continue
            try:
                with open(ep) as f:
                    ev = json.load(f)
            except Exception:
                continue
            pk = {}
            if pp.exists():
                try:
                    with open(pp) as f:
                        pk = json.load(f)
                except Exception:
                    pk = {}
            entry = splits.setdefault(sub.name, {"step_metrics": {}, "samples": {}})
            entry["step_metrics"][s] = {
                "response_metrics": pk.get("response_metrics", {}),
                "pass_at_k": pk.get("pass_at_k", {}),
                "k": pk.get("k"),
                "eval_data": pk.get("eval_data", ""),
            }
            for row in ev.get("rows", []):
                sid = str(row.get("id"))
                graders = row.get("graders") or []
                responses = row.get("coerced_responses") or row.get("responses") or []
                grades = [_grade_of(g) for g in graders]
                predicted = [str(g.get("predicted_answer", "")) if isinstance(g, dict) else "" for g in graders]
                correct = sum(1 for c in grades if c == "C")
                attempted = sum(1 for c in grades if c in ("C", "I"))
                k = len(grades) or 1
                samp = entry["samples"].setdefault(
                    sid,
                    {
                        "id": sid,
                        "question": row.get("question") or (row.get("extra_info") or {}).get("question") or "",
                        "answer": row.get("answer")
                        or (row.get("reward_model") or {}).get("ground_truth")
                        or (row.get("extra_info") or {}).get("answer")
                        or "",
                        "metadata": _to_jsonable(row.get("metadata") or (row.get("extra_info") or {}).get("metadata") or {}),
                        "per_step": {},
                    },
                )
                samp["per_step"][str(s)] = {
                    "k": k,
                    "correct": correct,
                    "attempted": attempted,
                    "pass_at_1": correct / k,
                    "pass_at_k": 1.0 if correct > 0 else 0.0,
                    "grades": "".join(grades),
                    "predicted": [_truncate(p, 200) for p in predicted],
                    "responses": [_truncate(r, resp_chars) for r in responses],
                }
    for split, entry in splits.items():
        entry["steps"] = sorted(entry["step_metrics"].keys())
    return splits


def load_sft_index(sft_parquet: Path | None, needed_ids: set[str], sft_chars: int):
    if sft_parquet is None:
        return {}
    import pandas as pd  # local import
    df = pd.read_parquet(sft_parquet)
    if "id" not in df.columns:
        return {}
    idx: dict[str, list] = {}
    for _, row in df.iterrows():
        sid = str(row["id"])
        if needed_ids and sid not in needed_ids:
            continue
        prompt = row.get("prompt")
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        prompt_text = ""
        if isinstance(prompt, list) and prompt:
            parts = []
            for m in prompt:
                if isinstance(m, dict):
                    parts.append(f"[{m.get('role','?')}] {m.get('content','')}")
            prompt_text = "\n".join(parts)
        else:
            prompt_text = str(prompt or "")
        extra = _to_jsonable(row.get("extra_info")) if row.get("extra_info") is not None else {}
        item = {
            "id": sid,
            "question": str(row.get("question") or ""),
            "answer": str(row.get("answer") or ""),
            "prompt": _truncate(prompt_text, sft_chars),
            "extra_info": extra,
        }
        idx.setdefault(sid, []).append(item)
    return idx


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><title>RL explorer — __EXP_NAME__</title>
<style>
:root{--bg:#0f1115;--fg:#e6e6e6;--muted:#9aa3b2;--panel:#171a21;--accent:#38bdf8;--c-correct:#22c55e;--c-incorrect:#ef4444;--c-not:#6b7280;--c-failed:#a855f7;}
html,body{background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,sans-serif;margin:0;}
header{padding:12px 18px;border-bottom:1px solid #222;display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;}
header h1{font-size:15px;margin:0;font-weight:600;}
header .meta{color:var(--muted);font-size:12px;font-family:ui-monospace,Menlo,monospace;}
.tabs{display:flex;gap:4px;padding:10px 18px 0 18px;background:var(--panel);border-bottom:1px solid #222;}
.tab{padding:6px 14px;font-size:13px;background:#0b0d12;border:1px solid #2a2f3a;border-bottom:none;border-radius:4px 4px 0 0;cursor:pointer;color:var(--muted);}
.tab.active{background:var(--accent);color:#000;border-color:var(--accent);font-weight:600;}
.controls{padding:10px 18px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;background:var(--panel);border-bottom:1px solid #222;}
.controls label{font-size:12px;color:var(--muted);}
.controls select,.controls input{background:#0b0d12;color:var(--fg);border:1px solid #2a2f3a;border-radius:4px;padding:5px 7px;font-size:12px;font-family:ui-monospace,Menlo,monospace;}
.controls input[type=text]{min-width:260px;}
main{padding:14px 18px;display:grid;grid-template-columns:1fr;gap:14px;}
.row{display:grid;grid-template-columns:minmax(420px,1fr) minmax(420px,1fr);gap:14px;}
.card{background:var(--panel);border:1px solid #222;border-radius:6px;padding:12px;}
.card h2{font-size:12px;margin:0 0 8px 0;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;}
table{width:100%;border-collapse:collapse;font-size:12px;font-family:ui-monospace,Menlo,monospace;}
th,td{padding:4px 6px;border-bottom:1px solid #222;text-align:left;vertical-align:top;}
th{color:var(--muted);font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:.05em;}
td.num,th.num{text-align:right;}
.q{background:#0b0d12;border:1px solid #222;border-radius:4px;padding:8px;white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#cbd5e1;}
.ans{color:var(--c-correct);font-weight:600;}
.pill{display:inline-block;padding:1px 6px;border-radius:3px;font-size:11px;font-family:ui-monospace,Menlo,monospace;margin-right:3px;}
.pill.C{background:#164e2b;color:#b8f0c8;}
.pill.I{background:#3a1414;color:#ff9a9a;}
.pill.N{background:#1f2430;color:#9ca3af;}
.pill.F{background:#2a1236;color:#d8b4fe;}
.pill._{background:#1f2937;color:#6b7280;}
.chart{width:100%;height:260px;background:#0b0d12;border:1px solid #222;border-radius:4px;}
.sample-list{max-height:560px;overflow:auto;border:1px solid #222;border-radius:4px;background:#0b0d12;}
.sample-list .item{padding:6px 10px;border-bottom:1px solid #1a1d24;cursor:pointer;font-size:12px;font-family:ui-monospace,Menlo,monospace;}
.sample-list .item:hover{background:#111520;}
.sample-list .item.active{background:#10263a;color:#fff;}
.sample-list .item .id{color:var(--accent);}
.sample-list .item .q{background:none;border:none;padding:0;color:#cbd5e1;max-height:2.6em;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;}
.resp-block{background:#0b0d12;border:1px solid #222;border-radius:4px;margin:3px 0;padding:6px 8px;font-family:ui-monospace,Menlo,monospace;font-size:11px;white-space:pre-wrap;color:#cbd5e1;}
.resp-block.C{border-left:3px solid var(--c-correct);}
.resp-block.I{border-left:3px solid var(--c-incorrect);}
.resp-block.N{border-left:3px solid var(--c-not);}
.resp-block.F{border-left:3px solid var(--c-failed);}
details{margin:4px 0;}
summary{cursor:pointer;color:var(--muted);font-size:11px;font-family:ui-monospace,Menlo,monospace;}
.muted{color:var(--muted);font-size:11px;}
.hidden{display:none;}
.axis{stroke:#555;stroke-width:1;}
.grid{stroke:#222;stroke-width:1;}
.tick{fill:#9aa3b2;font-family:ui-monospace,Menlo,monospace;font-size:10px;}
.linepath{fill:none;stroke:var(--accent);stroke-width:2;}
.linepath.sample{stroke:#f59e0b;stroke-width:2.5;}
.dot{fill:var(--accent);stroke:#000;}
.dot.sample{fill:#f59e0b;}
.sft{background:#0b0d12;border:1px solid #222;border-radius:4px;padding:8px;margin-top:4px;}
.sft .title{color:var(--accent);font-size:11px;font-family:ui-monospace,Menlo,monospace;margin-bottom:4px;}
</style></head><body>
<header>
  <h1>RL experiment explorer</h1>
  <span class="meta">__EXP_DIR__</span>
  <span class="meta">steps: __STEPS__</span>
  __SFT_META__
</header>
<div class="tabs">
  <div class="tab active" data-tab="eval">Eval</div>
  <div class="tab" data-tab="passk">Pass@k</div>
</div>
<div id="tab-eval" class="tabpane">
  <div class="controls">
    <label>Split <select id="eval-split"></select></label>
    <label>Trajectory
      <select id="eval-traj">
        <option value="all">all</option>
        <option value="improved">improved (wrong → correct)</option>
        <option value="regressed">regressed (correct → wrong)</option>
        <option value="same_correct">same correct</option>
        <option value="same_incorrect">same incorrect</option>
        <option value="other">other / mixed</option>
      </select>
    </label>
    <label>Sample id/text <input id="eval-search" type="text" placeholder="filter by id or question..."/></label>
    <span class="muted" id="eval-status"></span>
  </div>
  <main>
    <div class="row">
      <div class="card"><h2>Samples</h2><div id="eval-samples" class="sample-list"></div></div>
      <div class="card"><h2>Accuracy over steps (split avg + selected sample)</h2>
        <svg class="chart" id="eval-chart"></svg>
        <div id="eval-split-metrics" class="muted"></div>
      </div>
    </div>
    <div class="card" id="eval-detail"><h2>Sample detail</h2><div id="eval-detail-body" class="muted">Select a sample.</div></div>
  </main>
</div>
<div id="tab-passk" class="tabpane hidden">
  <div class="controls">
    <label>Split <select id="pk-split"></select></label>
    <label>Sample id/text <input id="pk-search" type="text" placeholder="filter by id or question..."/></label>
    <span class="muted" id="pk-status"></span>
  </div>
  <main>
    <div class="row">
      <div class="card"><h2>Samples</h2><div id="pk-samples" class="sample-list"></div></div>
      <div class="card"><h2>pass@1 / pass@k over steps</h2>
        <svg class="chart" id="pk-chart"></svg>
        <div id="pk-split-metrics" class="muted"></div>
      </div>
    </div>
    <div class="card" id="pk-detail"><h2>Sample detail</h2><div id="pk-detail-body" class="muted">Select a sample.</div></div>
  </main>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
(function(){
const DATA = JSON.parse(document.getElementById('payload').textContent);
const STEPS = DATA.steps;
const SFT = DATA.sft_index || {};

function el(tag, attrs, children){
  const e = document.createElement(tag);
  if(attrs) for(const k in attrs){ if(k==='class') e.className=attrs[k]; else if(k==='html') e.innerHTML=attrs[k]; else e.setAttribute(k, attrs[k]); }
  (children||[]).forEach(c=>e.appendChild(typeof c==='string'?document.createTextNode(c):c));
  return e;
}
function esc(s){ return (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'})[c]); }

// ---- Tabs ----
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    t.classList.add('active');
    document.querySelectorAll('.tabpane').forEach(p=>p.classList.add('hidden'));
    document.getElementById('tab-'+t.dataset.tab).classList.remove('hidden');
  });
});

// ---- Line chart ----
function drawChart(svg, steps, seriesList, yDomain){
  svg.innerHTML='';
  const W = svg.clientWidth||600, H = svg.clientHeight||260;
  const L=40,R=12,T=10,B=26;
  const ns='http://www.w3.org/2000/svg';
  const xmin=Math.min(...steps), xmax=Math.max(...steps);
  const ymin=yDomain[0], ymax=yDomain[1];
  const xs = x => L + (W-L-R)*((x-xmin)/Math.max(1,(xmax-xmin)));
  const ys = y => T + (H-T-B)*(1 - (y-ymin)/Math.max(1e-9,(ymax-ymin)));
  // grid
  for(let i=0;i<=5;i++){
    const v = ymin + (ymax-ymin)*i/5;
    const y = ys(v);
    const l = document.createElementNS(ns,'line');
    l.setAttribute('x1',L);l.setAttribute('x2',W-R);l.setAttribute('y1',y);l.setAttribute('y2',y);l.setAttribute('class','grid');
    svg.appendChild(l);
    const t = document.createElementNS(ns,'text');
    t.setAttribute('x',L-4);t.setAttribute('y',y+3);t.setAttribute('text-anchor','end');t.setAttribute('class','tick');
    t.textContent = v.toFixed(2); svg.appendChild(t);
  }
  steps.forEach(s=>{
    const x=xs(s);
    const t=document.createElementNS(ns,'text');
    t.setAttribute('x',x);t.setAttribute('y',H-B+14);t.setAttribute('text-anchor','middle');t.setAttribute('class','tick');
    t.textContent=s; svg.appendChild(t);
  });
  seriesList.forEach(series=>{
    const pts = series.points.filter(p=>p[1]!=null);
    if(!pts.length) return;
    const d = pts.map((p,i)=>(i?'L':'M')+xs(p[0])+' '+ys(p[1])).join(' ');
    const path = document.createElementNS(ns,'path');
    path.setAttribute('d',d);path.setAttribute('class','linepath'+(series.cls?' '+series.cls:''));
    if(series.color) path.setAttribute('stroke',series.color);
    svg.appendChild(path);
    pts.forEach(p=>{
      const c=document.createElementNS(ns,'circle');
      c.setAttribute('cx',xs(p[0]));c.setAttribute('cy',ys(p[1]));c.setAttribute('r',3);
      c.setAttribute('class','dot'+(series.cls?' '+series.cls:''));
      if(series.color) c.setAttribute('fill',series.color);
      const title=document.createElementNS(ns,'title');
      title.textContent = series.name+' @step '+p[0]+' = '+p[1].toFixed(3);
      c.appendChild(title);
      svg.appendChild(c);
    });
  });
  // legend
  let lx=L+4;
  seriesList.forEach(s=>{
    const r=document.createElementNS(ns,'rect');
    r.setAttribute('x',lx);r.setAttribute('y',T);r.setAttribute('width',10);r.setAttribute('height',10);
    r.setAttribute('fill', s.color || (s.cls==='sample'?'#f59e0b':'#38bdf8'));
    svg.appendChild(r);
    const t=document.createElementNS(ns,'text');
    t.setAttribute('x',lx+14);t.setAttribute('y',T+9);t.setAttribute('class','tick');
    t.textContent=s.name; svg.appendChild(t);
    lx += 24 + s.name.length*6;
  });
}

// ---- SFT block ----
function sftBlock(sid){
  const matches = SFT[sid] || [];
  if(!matches.length) return el('div',{class:'muted'},['No SFT train row with id='+sid]);
  const box = el('div', {});
  box.appendChild(el('div',{class:'muted'},['SFT train matches by id ('+matches.length+')']));
  matches.forEach((m,i)=>{
    const d = el('details',{});
    d.appendChild(el('summary',{},['['+i+'] question: '+(m.question||'').slice(0,120)]));
    const body = el('div',{class:'sft'});
    body.appendChild(el('div',{class:'title'},['id='+m.id+' · answer='+m.answer]));
    body.appendChild(el('div',{class:'q'},[m.prompt]));
    const ex = el('details',{});
    ex.appendChild(el('summary',{},['extra_info']));
    ex.appendChild(el('pre',{class:'resp-block'},[JSON.stringify(m.extra_info,null,2)]));
    body.appendChild(ex);
    d.appendChild(body);
    box.appendChild(d);
  });
  return box;
}

// ---- Trajectory classification (eval mode) ----
// A sample's per-step acc is correct/k. We call a step "correct" if acc >= 0.5
// (majority correct for k>1; acc==1 for k=1). The trajectory compares the
// first and last step that have data for that sample.
function classifyTrajectory(samp){
  const withData = STEPS.filter(s => samp.per_step[String(s)]);
  if(withData.length < 2) return 'other';
  const first = samp.per_step[String(withData[0])];
  const last = samp.per_step[String(withData[withData.length-1])];
  const firstCorrect = (first.acc != null ? first.acc : first.pass_at_1) >= 0.5;
  const lastCorrect = (last.acc != null ? last.acc : last.pass_at_1) >= 0.5;
  if(firstCorrect && lastCorrect) return 'same_correct';
  if(!firstCorrect && !lastCorrect) return 'same_incorrect';
  if(!firstCorrect && lastCorrect) return 'improved';
  if(firstCorrect && !lastCorrect) return 'regressed';
  return 'other';
}
const TRAJ_LABEL = {
  improved: 'IMP',
  regressed: 'REG',
  same_correct: 'SC',
  same_incorrect: 'SI',
  other: '—',
};
const TRAJ_COLOR = {
  improved: '#22c55e',
  regressed: '#ef4444',
  same_correct: '#60a5fa',
  same_incorrect: '#6b7280',
  other: '#a855f7',
};

// ---- Generic tab renderer ----
function setupTab(cfg){
  const {data, splitSel, trajSel, searchInp, samplesDiv, statusSpan, chartSvg, splitMetricsDiv, detailBody, mode} = cfg;
  const splits = Object.keys(data).sort();
  splitSel.innerHTML='';
  splits.forEach(s=>{ const o=document.createElement('option'); o.value=s; o.textContent=s; splitSel.appendChild(o); });
  if(!splits.length){
    statusSpan.textContent='no data';
    return;
  }
  let currentSplit = splits[0];
  let currentSample = null;

  function sampleAccSeries(samp){
    const pts = STEPS.map(s=>{
      const ps = samp.per_step[String(s)];
      if(!ps) return [s,null];
      if(mode==='eval') return [s, ps.acc];
      return [s, ps.pass_at_1];
    });
    return pts;
  }
  function samplePassKSeries(samp){
    return STEPS.map(s=>{
      const ps = samp.per_step[String(s)];
      return [s, ps? ps.pass_at_k : null];
    });
  }
  function splitAvgSeries(entry){
    const sids = Object.keys(entry.samples);
    return STEPS.map(s=>{
      let tot=0,n=0;
      sids.forEach(id=>{
        const ps = entry.samples[id].per_step[String(s)];
        if(ps){ tot += (mode==='eval'? ps.acc : ps.pass_at_1); n++; }
      });
      return [s, n? tot/n : null];
    });
  }
  function splitAvgPassK(entry){
    const sids = Object.keys(entry.samples);
    return STEPS.map(s=>{
      let tot=0,n=0;
      sids.forEach(id=>{
        const ps = entry.samples[id].per_step[String(s)];
        if(ps){ tot += ps.pass_at_k; n++; }
      });
      return [s, n? tot/n : null];
    });
  }

  function renderChart(){
    const entry = data[currentSplit];
    const series = [];
    series.push({name: mode==='eval'?'split avg acc':'split avg pass@1', points: splitAvgSeries(entry), color:'#38bdf8'});
    if(mode==='passk') series.push({name:'split avg pass@k', points: splitAvgPassK(entry), color:'#22c55e'});
    if(currentSample && entry.samples[currentSample]){
      series.push({name:'sample '+currentSample+(mode==='eval'?' acc':' pass@1'), points: sampleAccSeries(entry.samples[currentSample]), cls:'sample'});
      if(mode==='passk') series.push({name:'sample pass@k', points: samplePassKSeries(entry.samples[currentSample]), color:'#f59e0b'});
    }
    drawChart(chartSvg, STEPS, series, [0,1]);
  }

  function renderSplitMetrics(){
    const entry = data[currentSplit];
    const rows = STEPS.map(s=>{
      const m = entry.step_metrics[s];
      if(!m) return null;
      if(mode==='eval'){
        return [s, m.accuracy, m.attempt_rate, m.correct, m.incorrect, m.not_attempted, m.failed_to_parse, m.total_responses];
      } else {
        const rm = m.response_metrics||{};
        const pk = m.pass_at_k||{};
        const kv = Object.keys(pk).sort((a,b)=>(+a)-(+b));
        return [s, rm.accuracy, rm.attempt_rate, kv.map(k=>'pass@'+k+'='+(+pk[k]).toFixed(3)).join(' ')];
      }
    }).filter(Boolean);
    const tbl = el('table',{});
    const thead = el('tr',{});
    if(mode==='eval') ['step','acc','attempt','C','I','N','F','total'].forEach(h=>thead.appendChild(el('th',{class:'num'},[h])));
    else ['step','acc','attempt','pass@k'].forEach(h=>thead.appendChild(el('th',{},[h])));
    tbl.appendChild(thead);
    rows.forEach(r=>{
      const tr = el('tr',{});
      r.forEach((v,i)=>{
        const td = el('td',{class: (typeof v==='number'?'num':'')},[typeof v==='number'?v.toFixed(3):(v==null?'':String(v))]);
        tr.appendChild(td);
      });
      tbl.appendChild(tr);
    });
    splitMetricsDiv.innerHTML='';
    splitMetricsDiv.appendChild(el('div',{class:'muted'},['Per-step split metrics']));
    splitMetricsDiv.appendChild(tbl);
  }

  function renderSamples(){
    const entry = data[currentSplit];
    const q = (searchInp.value||'').toLowerCase().trim();
    const trajFilter = trajSel ? trajSel.value : 'all';
    const sids = Object.keys(entry.samples).sort((a,b)=>{
      const na = +a, nb = +b;
      if(!Number.isNaN(na)&&!Number.isNaN(nb)) return na-nb;
      return a.localeCompare(b);
    });
    // classify all samples once so we can also show counts
    const trajCounts = {improved:0, regressed:0, same_correct:0, same_incorrect:0, other:0};
    const trajBySid = {};
    sids.forEach(id=>{
      const tr = classifyTrajectory(entry.samples[id]);
      trajBySid[id] = tr;
      trajCounts[tr] = (trajCounts[tr]||0) + 1;
    });
    samplesDiv.innerHTML='';
    let shown = 0, matchedTraj = 0;
    sids.forEach(id=>{
      const s = entry.samples[id];
      const tr = trajBySid[id];
      if(trajFilter !== 'all' && tr !== trajFilter) return;
      matchedTraj++;
      if(q){
        const hay = (id+' '+(s.question||'')+' '+(s.answer||'')).toLowerCase();
        if(hay.indexOf(q)<0) return;
      }
      shown++;
      if(shown>500) return;
      const item = el('div',{class:'item'+(id===currentSample?' active':'')});
      // tiny sparkline of acc
      const accs = STEPS.map(st=>{ const ps = s.per_step[String(st)]; return ps? (mode==='eval'?ps.acc:ps.pass_at_1):null; });
      const avg = (accs.filter(x=>x!=null).reduce((a,b)=>a+b,0)/Math.max(1,accs.filter(x=>x!=null).length));
      const trajBadge = el('span',{class:'pill', style:'background:'+TRAJ_COLOR[tr]+';color:#000;font-weight:600;'},[TRAJ_LABEL[tr]]);
      item.appendChild(el('div',{},[el('span',{class:'id'},['#'+id]),' ', trajBadge, ' avg='+avg.toFixed(2)+' · has_sft='+((SFT[id]||[]).length>0?'yes('+(SFT[id]||[]).length+')':'no')]));
      item.appendChild(el('div',{class:'q'},[s.question||'(no question)']));
      item.addEventListener('click',()=>{ currentSample=id; renderAll(); });
      samplesDiv.appendChild(item);
    });
    const trajSummary = 'IMP='+trajCounts.improved+' REG='+trajCounts.regressed+' SC='+trajCounts.same_correct+' SI='+trajCounts.same_incorrect+' OTH='+trajCounts.other;
    statusSpan.textContent = shown+' shown / '+matchedTraj+' matched / '+sids.length+' total'+(shown>500?' (truncated to 500)':'')+' · '+trajSummary;
  }

  function renderDetail(){
    detailBody.innerHTML='';
    if(!currentSample){ detailBody.appendChild(el('div',{class:'muted'},['Select a sample from the list.'])); return; }
    const entry = data[currentSplit];
    const samp = entry.samples[currentSample];
    if(!samp){ detailBody.appendChild(el('div',{class:'muted'},['Sample not found in this split.'])); return; }
    const head = el('div',{});
    head.appendChild(el('div',{},[el('span',{class:'ans'},['id='+samp.id]),' · gold answer: ', el('span',{class:'ans'},[samp.answer||''])]));
    head.appendChild(el('div',{class:'q'},[samp.question||'']));
    const meta = samp.metadata||{};
    if(Object.keys(meta).length){
      const md = el('details',{});
      md.appendChild(el('summary',{},['metadata']));
      md.appendChild(el('pre',{class:'resp-block'},[JSON.stringify(meta,null,2)]));
      head.appendChild(md);
    }
    head.appendChild(el('div',{class:'muted'},['\u2014 SFT train match \u2014']));
    head.appendChild(sftBlock(samp.id));
    detailBody.appendChild(head);

    // per-step table
    const tbl = el('table',{});
    const thead = el('tr',{});
    if(mode==='eval') ['step','k','correct','acc','grades'].forEach(h=>thead.appendChild(el('th',{},[h])));
    else ['step','k','correct','pass@1','pass@k','grades'].forEach(h=>thead.appendChild(el('th',{},[h])));
    tbl.appendChild(thead);
    STEPS.forEach(st=>{
      const ps = samp.per_step[String(st)];
      if(!ps) return;
      const tr = el('tr',{});
      if(mode==='eval'){
        tr.appendChild(el('td',{},[String(st)]));
        tr.appendChild(el('td',{class:'num'},[String(ps.k)]));
        tr.appendChild(el('td',{class:'num'},[String(ps.correct)]));
        tr.appendChild(el('td',{class:'num'},[ps.acc.toFixed(3)]));
      } else {
        tr.appendChild(el('td',{},[String(st)]));
        tr.appendChild(el('td',{class:'num'},[String(ps.k)]));
        tr.appendChild(el('td',{class:'num'},[String(ps.correct)]));
        tr.appendChild(el('td',{class:'num'},[ps.pass_at_1.toFixed(3)]));
        tr.appendChild(el('td',{class:'num'},[ps.pass_at_k.toFixed(0)]));
      }
      const gcell = el('td',{});
      (ps.grades||'').split('').forEach(g=>{
        gcell.appendChild(el('span',{class:'pill '+(g||'_')},[g||'?']));
      });
      tr.appendChild(gcell);
      tbl.appendChild(tr);
    });
    detailBody.appendChild(el('h3',{},['Per-step grades']));
    detailBody.appendChild(tbl);

    // per-step responses
    detailBody.appendChild(el('h3',{},['Per-step responses (click to expand)']));
    STEPS.forEach(st=>{
      const ps = samp.per_step[String(st)];
      if(!ps) return;
      const det = el('details',{});
      const grades = (ps.grades||'');
      det.appendChild(el('summary',{},['step '+st+' · k='+ps.k+' · C='+ps.correct+' · ['+grades+']']));
      (ps.responses||[]).forEach((r,idx)=>{
        const g = grades[idx]||'_';
        const pred = (ps.predicted||[])[idx]||'';
        const block = el('div',{class:'resp-block '+g});
        block.appendChild(el('div',{class:'muted'},['['+idx+'] grade='+g+' · predicted="'+pred+'"']));
        block.appendChild(document.createTextNode(r||'(response truncated/empty)'));
        det.appendChild(block);
      });
      detailBody.appendChild(det);
    });
  }

  function renderAll(){ renderChart(); renderSplitMetrics(); renderSamples(); renderDetail(); }

  splitSel.addEventListener('change',()=>{ currentSplit=splitSel.value; currentSample=null; renderAll(); });
  searchInp.addEventListener('input',()=>{ renderSamples(); });
  if(trajSel) trajSel.addEventListener('change',()=>{ renderSamples(); });
  renderAll();
}

setupTab({
  data: DATA.eval,
  splitSel: document.getElementById('eval-split'),
  trajSel: document.getElementById('eval-traj'),
  searchInp: document.getElementById('eval-search'),
  samplesDiv: document.getElementById('eval-samples'),
  statusSpan: document.getElementById('eval-status'),
  chartSvg: document.getElementById('eval-chart'),
  splitMetricsDiv: document.getElementById('eval-split-metrics'),
  detailBody: document.getElementById('eval-detail-body'),
  mode: 'eval',
});
setupTab({
  data: DATA.passk,
  splitSel: document.getElementById('pk-split'),
  trajSel: null,
  searchInp: document.getElementById('pk-search'),
  samplesDiv: document.getElementById('pk-samples'),
  statusSpan: document.getElementById('pk-status'),
  chartSvg: document.getElementById('pk-chart'),
  splitMetricsDiv: document.getElementById('pk-split-metrics'),
  detailBody: document.getElementById('pk-detail-body'),
  mode: 'passk',
});
})();
</script>
</body></html>
"""


def build_html(exp_dir: Path, sft_parquet: Path | None, resp_chars: int, sft_chars: int) -> str:
    steps = discover_steps(exp_dir)
    eval_data = load_eval(exp_dir, steps, resp_chars)
    passk_data = load_pass_at_k(exp_dir, steps, resp_chars)

    needed_ids: set[str] = set()
    for entry in eval_data.values():
        needed_ids.update(entry["samples"].keys())
    for entry in passk_data.values():
        needed_ids.update(entry["samples"].keys())

    sft_index = load_sft_index(sft_parquet, needed_ids, sft_chars)

    # normalize step_metrics keys to strings for JSON
    def _normalize(splits):
        out = {}
        for k, v in splits.items():
            out[k] = {
                "steps": v["steps"],
                "step_metrics": {str(s): v["step_metrics"][s] for s in v["step_metrics"]},
                "samples": v["samples"],
            }
        return out

    payload = {
        "exp_dir": str(exp_dir),
        "steps": steps,
        "eval": _normalize(eval_data),
        "passk": _normalize(passk_data),
        "sft_index": sft_index,
    }
    raw = json.dumps(payload, ensure_ascii=False, default=str)
    raw = raw.replace("</script", "<\\/script")

    sft_meta = (
        f'<span class="meta">sft: {_html.escape(str(sft_parquet))} (matches: {len(sft_index)})</span>'
        if sft_parquet else '<span class="meta">sft: (none provided)</span>'
    )
    html = (HTML_TEMPLATE
            .replace("__EXP_NAME__", _html.escape(exp_dir.name))
            .replace("__EXP_DIR__", _html.escape(str(exp_dir)))
            .replace("__STEPS__", _html.escape(",".join(str(s) for s in steps)))
            .replace("__SFT_META__", sft_meta)
            .replace("__PAYLOAD__", raw))
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True, type=Path)
    ap.add_argument("--sft-train", type=Path, default=None, help="Optional SFT train parquet for id-based join")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resp-chars", type=int, default=2000, help="Max chars per response (set 0 to omit)")
    ap.add_argument("--sft-chars", type=int, default=4000, help="Max chars per SFT prompt")
    args = ap.parse_args()

    exp_dir = args.exp_dir.resolve()
    if not exp_dir.exists():
        raise SystemExit(f"exp-dir not found: {exp_dir}")
    out = args.out or (exp_dir / "explorer.html")
    html = build_html(exp_dir, args.sft_train.resolve() if args.sft_train else None, args.resp_chars, args.sft_chars)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
