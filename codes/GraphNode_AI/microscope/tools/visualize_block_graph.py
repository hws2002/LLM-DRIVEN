#!/usr/bin/env python3
"""Block Graph Visualizer.

Usage:
    python microscope/tools/visualize_block_graph.py <block_graph.json>
"""
import argparse
import json
import re
import sys
import webbrowser
from pathlib import Path

# Block-level edge colors
BLOCK_EDGE_COLORS = {
    "PREREQUISITE_OF": "#ef4444",
    "FOLLOWS":         "#38bdf8",
    "ELABORATES":      "#22c55e",
    "CONTRASTS":       "#f97316",
    "PARALLEL":        "#94a3b8",
}

# Micro-level edge colors (from type_mapping.json)
MICRO_EDGE_COLORS = {
    "defines":         "#a78bfa",
    "uses":            "#14b8a6",
    "part_of":         "#2dd4bf",
    "causes":          "#f97316",
    "influences":      "#f472b6",
    "leads_to":        "#fb7185",
    "solves":          "#34d399",
    "improves":        "#4ade80",
    "requires":        "#ef4444",
    "prerequisite_of": "#34d399",
    "derives_from":    "#60a5fa",
    "equivalent_to":   "#c084fc",
    "example_of":      "#fbbf24",
    "contrasts_with":  "#60a5fa",
    "applied_in":      "#38bdf8",
    "occurs_in":       "#94a3b8",
    "happens_during":  "#94a3b8",
    "proposed_by":     "#fbbf24",
}

KO_MAP = {
    # block-level
    "PREREQUISITE_OF": "text",
    "FOLLOWS":         "text",
    "ELABORATES":      "text",
    "CONTRASTS":       "text",
    "PARALLEL":        "text",
    # micro-level
    "defines":         "text",
    "uses":            "text",
    "part_of":         "text",
    "causes":          "text",
    "influences":      "text",
    "leads_to":        "text",
    "solves":          "text",
    "improves":        "text",
    "requires":        "text",
    "prerequisite_of": "text",
    "derives_from":    "text",
    "equivalent_to":   "text",
    "example_of":      "text",
    "contrasts_with":  "text",
    "applied_in":      "text",
    "occurs_in":       "text",
    "happens_during":  "text",
    "proposed_by":     "Proposed by",
}


def _load_turn_texts(src: Path) -> dict[int, str]:
    prompt_path = src.parent / "01_segmentation" / "prompt_user.txt"
    if not prompt_path.exists():
        return {}

    text = prompt_path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^\[Turn\s+(\d+)\]\s*", text))
    turns: dict[int, str] = {}
    for idx, match in enumerate(matches):
        turn = int(match.group(1))
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        turns[turn] = text[start:end].strip()
    return turns


def _raw_text_from_turn_range(block: dict, turn_texts: dict[int, str]) -> str:
    turn_range = block.get("turn_range")
    if not isinstance(turn_range, list) or len(turn_range) != 2:
        return ""

    start, end = turn_range
    if not isinstance(start, int) or not isinstance(end, int):
        return ""

    return "\n\n".join(
        turn_texts[turn]
        for turn in range(start, end + 1)
        if turn in turn_texts
    )


def load_raw_texts(src: Path, blocks: list) -> dict:
    micro_dir = src.parent / "03_micro_graphs"
    block_text_dir = src.parent / "03_block_texts"
    turn_texts = _load_turn_texts(src)
    result = {}
    for b in blocks:
        bid = b["block_id"]
        micro_raw = micro_dir / bid / "raw_text.txt"
        block_raw = block_text_dir / bid / "raw_text.txt"
        if micro_raw.exists():
            result[bid] = micro_raw.read_text(encoding="utf-8")
        elif block_raw.exists():
            result[bid] = block_raw.read_text(encoding="utf-8")
        else:
            result[bid] = b.get("raw_text") or _raw_text_from_turn_range(b, turn_texts)
    return result


def safe_json(obj) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("</", "<\\/")


def normalize_blocks(blocks: list) -> list:
    """Add id/label fields to micro_graph nodes so JS can reference them uniformly."""
    out = []
    for b in blocks:
        b = dict(b)
        mg = b.get("micro_graph") or {}
        nodes = mg.get("nodes") or []
        edges = mg.get("edges") or []
        new_nodes = []
        for i, n in enumerate(nodes):
            n = dict(n)
            n.setdefault("id", f"{b['block_id']}::n{i}")
            n.setdefault("label", n.get("name", ""))
            new_nodes.append(n)
        # edges use 'start'/'target' — normalize to 'source'/'target' for JS
        id_by_name = {n.get("name", ""): n["id"] for n in new_nodes}
        new_edges = []
        for e in edges:
            e = dict(e)
            src_name = e.pop("start", None)
            if src_name is not None:
                e["source"] = id_by_name.get(src_name, src_name)
            tgt_name = e.get("target")
            if tgt_name is not None:
                e["target"] = id_by_name.get(tgt_name, tgt_name)
            new_edges.append(e)
        b["micro_graph"] = {"nodes": new_nodes, "edges": new_edges}
        out.append(b)
    return out


def collect_colors(blocks: list, edges: list) -> dict:
    colors = dict(BLOCK_EDGE_COLORS)
    colors.update(MICRO_EDGE_COLORS)
    for e in edges:
        t = e.get("type", "")
        if t and t not in colors:
            colors[t] = "#94a3b8"
    for b in blocks:
        for e in (b.get("micro_graph") or {}).get("edges") or []:
            t = e.get("type", "")
            if t and t not in colors:
                colors[t] = "#94a3b8"
    return colors


def build_html(data: dict, raw_texts: dict, src: Path) -> str:
    bg          = data.get("block_graph", {})
    blocks_raw  = bg.get("blocks", [])
    edges       = bg.get("edges", [])
    paths       = bg.get("paths", [])
    source_name = data.get("source_name", src.name)

    blocks = normalize_blocks(blocks_raw)
    colors = collect_colors(blocks, edges)

    total_micro_nodes = sum(len((b.get("micro_graph") or {}).get("nodes") or []) for b in blocks)
    total_micro_edges = sum(len((b.get("micro_graph") or {}).get("edges") or []) for b in blocks)
    meta = f"{len(blocks)} blocks · {len(edges)} block edges · {total_micro_nodes} micro nodes · {total_micro_edges} micro edges"

    blocks_j = safe_json(blocks)
    edges_j  = safe_json(edges)
    paths_j  = safe_json(paths)
    colors_j = safe_json(colors)
    ko_j     = safe_json(KO_MAP)
    raw_j    = safe_json(raw_texts)

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{source_name} - block graph</title>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['\\\\(', '\\\\)'], ['$', '$']] }},
  svg: {{ fontCache: 'global' }},
  startup: {{ typeset: false }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
* {{ box-sizing:border-box; }}
html, body {{ width:100%; height:100%; margin:0; overflow:hidden; }}
body {{ font-family:Inter,Segoe UI,Arial,sans-serif; background:#0b0f1a; color:#e5e7eb; }}
#topbar {{ height:54px; display:flex; align-items:center; gap:18px; padding:0 18px; background:#111827; border-bottom:1px solid #263044; }}
#title {{ font-weight:750; font-size:15px; }}
#meta {{ color:#9ca3af; font-size:12px; }}
#legend {{ margin-left:auto; display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end; }}
.legend-item {{ display:flex; align-items:center; gap:5px; color:#aab2c5; font-size:11px; }}
.legend-line {{ width:18px; height:3px; border-radius:999px; }}
#stage {{ position:relative; height:calc(100vh - 54px); overflow:hidden; cursor:grab; }}
#world {{ position:absolute; inset:0; transform-origin:0 0; }}
#edge-svg {{ position:absolute; inset:0; width:3600px; height:2400px; pointer-events:none; overflow:visible; }}
#block-layer {{ position:absolute; inset:0; }}
.block-card {{ position:absolute; width:330px; height:286px; background:#202738; border:1px solid #55627a; border-radius:8px; overflow:hidden; box-shadow:0 18px 38px #0008; cursor:grab; transition:opacity .18s; }}
.block-card:active {{ cursor:grabbing; }}
.block-card.focused {{ border-color:#cbd5e1; box-shadow:0 0 0 2px #e5e7eb33, 0 18px 42px #000a; }}
.block-card.dimmed {{ opacity:0.12; pointer-events:none; }}
.block-card.highlighted {{ border-color:#60a5fa; box-shadow:0 0 0 2px #60a5fa55, 0 18px 42px #000a; opacity:1; }}
#path-btns {{ display:flex; gap:6px; flex-wrap:wrap; align-items:center; flex:0 0 auto; }}
.path-title {{ color:#e5e7eb; font-size:12px; font-weight:700; margin-right:2px; }}
.path-btn {{ padding:4px 12px; border-radius:6px; border:1px solid #475569; background:#1f2937; color:#e5e7eb; font-size:12px; cursor:pointer; }}
.path-btn:hover {{ background:#374151; color:#e5e7eb; }}
.path-btn.active {{ background:#2563eb; border-color:#3b82f6; color:#fff; }}
#reset-btn {{ padding:4px 10px; border-radius:6px; border:1px solid #374151; background:#1f2937; color:#6b7280; font-size:12px; cursor:pointer; display:none; }}
#reset-btn:hover {{ color:#e5e7eb; }}
.head {{ height:78px; padding:14px 16px 10px; background:#151b28; user-select:none; }}
.block-id {{ font:12px Consolas,monospace; color:#7d8798; margin-bottom:6px; }}
.block-title {{ color:#f8fafc; font-weight:800; font-size:17px; line-height:1.25; max-height:42px; overflow:hidden; }}
.micro {{ position:relative; height:166px; background:#252d3f; border-top:1px solid #2f394d; border-bottom:1px solid #2f394d; cursor:pointer; }}
.micro svg {{ width:100%; height:100%; display:block; }}
.micro-empty {{ height:166px; display:flex; align-items:center; justify-content:center; color:#7d8798; font-size:12px; background:#252d3f; }}
.micro-node-label {{ width:86px; max-height:26px; color:#e5e7eb; font-size:9px; line-height:1.12; text-align:center; overflow:hidden; }}
.foot {{ min-height:42px; padding:8px 12px; display:flex; flex-wrap:wrap; align-content:flex-start; gap:5px; background:#171e2c; }}
.tag {{ max-width:145px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; padding:3px 8px; border-radius:999px; background:#374151; color:#d1d5db; font-size:12px; }}
.edge-label {{ font-size:11px; font-weight:700; paint-order:stroke; stroke:#0b0f1a; stroke-width:4px; }}
.micro-label {{ font-size:9px; fill:#e5e7eb; paint-order:stroke; stroke:#252d3f; stroke-width:3px; pointer-events:none; }}
.micro-node {{ fill:#d8dee9; stroke:#111827; stroke-width:1.2; }}
.hint {{ position:absolute; left:14px; bottom:12px; color:#7d8798; font-size:12px; pointer-events:none; }}
#raw-panel {{ position:absolute; top:14px; right:14px; width:420px; max-height:calc(100% - 28px); display:none; background:#111827f2; border:1px solid #3b475e; border-radius:8px; box-shadow:0 18px 44px #000b; overflow:hidden; z-index:20; }}
#raw-title {{ padding:12px 42px 8px 14px; font-weight:800; color:#f8fafc; background:#151b28; border-bottom:1px solid #2f394d; position:relative; }}
#raw-close {{ position:absolute; top:8px; right:10px; border:1px solid #4b5563; background:#1f2937; color:#f8fafc; border-radius:5px; padding:2px 7px; cursor:pointer; }}
#raw-body {{ padding:13px 15px 16px; color:#dbe2ef; font-size:13px; line-height:1.65; overflow:auto; max-height:calc(100vh - 126px); }}
#raw-body h1, #raw-body h2, #raw-body h3, #raw-body h4 {{ margin:14px 0 8px; color:#f8fafc; line-height:1.25; }}
#raw-body h1 {{ font-size:20px; }} #raw-body h2 {{ font-size:18px; }} #raw-body h3 {{ font-size:16px; }} #raw-body h4 {{ font-size:14px; }}
#raw-body p {{ margin:9px 0; }}
#raw-body ul {{ margin:8px 0 8px 20px; padding:0; }}
#raw-body li {{ margin:5px 0; }}
#raw-body strong {{ color:#ffffff; font-weight:800; }}
#raw-body hr {{ border:0; border-top:1px solid #334155; margin:14px 0; }}
#raw-body pre {{ white-space:pre-wrap; overflow:auto; background:#0b1020; border:1px solid #273244; border-radius:6px; padding:9px; }}
#raw-body .math-block {{ margin:10px 0; overflow-x:auto; }}
#micro-view {{ position:absolute; inset:0; display:none; background:#0b0f1af5; z-index:30; }}
#micro-toolbar {{ height:54px; display:flex; align-items:center; gap:14px; padding:0 16px; background:#111827; border-bottom:1px solid #2f394d; }}
#back-btn {{ border:1px solid #4b5563; background:#1f2937; color:#f8fafc; border-radius:6px; padding:7px 12px; cursor:pointer; }}
#micro-title {{ font-weight:800; }}
#micro-canvas {{ width:100%; height:calc(100% - 54px); }}
#micro-canvas svg {{ width:100%; height:100%; display:block; }}
#micro-canvas .big-node {{ fill:#d8dee9; stroke:#111827; stroke-width:2; }}
#micro-canvas .big-label {{ fill:#f8fafc; font-size:14px; paint-order:stroke; stroke:#0b0f1a; stroke-width:5px; }}
.big-node-label {{ width:230px; max-height:54px; color:#f8fafc; font-size:14px; line-height:1.2; text-align:center; overflow:hidden; }}
.big-node-group {{ cursor:grab; }}
.big-node-group:active {{ cursor:grabbing; }}
</style>
</head>
<body>
<div id="topbar">
  <div id="title">{source_name}</div>
  <div id="meta">{meta}</div>
  <div id="path-btns"></div>
  <button id="reset-btn" onclick="resetHighlight()">Reset</button>
  <div id="legend"></div>
</div>
<div id="stage">
  <div id="world">
    <svg id="edge-svg"><defs id="defs"></defs><g id="edges"></g><g id="labels"></g></svg>
    <div id="block-layer"></div>
  </div>
  <div class="hint">text text text text · text text · text text text text · text text text text</div>
  <div id="raw-panel">
    <div id="raw-title"><button id="raw-close">✕</button><span id="raw-title-text"></span></div>
    <div id="raw-body"></div>
  </div>
  <div id="micro-view">
    <div id="micro-toolbar">
      <button id="back-btn">← Close</button>
      <div id="micro-title"></div>
    </div>
    <div id="micro-canvas"></div>
  </div>
</div>

<script>
const BLOCKS = {blocks_j};
const EDGES  = {edges_j};
const PATHS  = {paths_j};
const COLORS = {colors_j};
const KO     = {ko_j};
const RAW_TEXTS = {raw_j};
const BLOCK_W = 330, BLOCK_H = 286;
let pos = {{}};
let panX = 20, panY = 20, scale = 1;
let panning = false, panStartX = 0, panStartY = 0;
let rawPinned = false;
let activePathIdx = null;

function escapeHtml(s) {{
  return String(s || '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function texify(s) {{
  s = String(s || '');
  const esc = escapeHtml(s);
  if (s.includes('\\\\(') || s.includes('\\\\[')) return esc;
  const hasMath = /\\\\(frac|int|sum|prod|sin|cos|tan|sec|mathbb|left|right)|[$_^=]/.test(s) && !/[\uAC00-\uD7A3A-Za-z]{{2,}}\\s+[\uAC00-\uD7A3A-Za-z]{{2,}}/.test(s);
  return hasMath ? '\\\\(' + esc + '\\\\)' : esc;
}}
function edgeLabel(t) {{ return KO[t] || t || ''; }}
function trunc(s, n) {{ s = String(s || ''); return s.length > n ? s.slice(0, n-3) + '...' : s; }}
function cssId(s) {{ return String(s || 'x').replace(/[^a-zA-Z0-9_-]/g, '_'); }}
function typeset() {{ if (window.MathJax?.typesetPromise) MathJax.typesetPromise().catch(()=>{{}}); }}

function layout() {{
  const ids = BLOCKS.map(b => b.block_id);
  const indeg = Object.fromEntries(ids.map(id => [id, 0]));
  const adj   = Object.fromEntries(ids.map(id => [id, []]));
  EDGES.forEach(e => {{
    if (adj[e.source]) {{ adj[e.source].push(e.target); indeg[e.target] = (indeg[e.target]||0)+1; }}
  }});
  const q = ids.filter(id => !indeg[id]);
  const level = Object.fromEntries(q.map(id => [id, 0]));
  for (let qi = 0; qi < q.length; qi++) {{
    const cur = q[qi];
    (adj[cur]||[]).forEach(n => {{
      level[n] = Math.max(level[n]||0, (level[cur]||0)+1);
      if (--indeg[n] === 0) q.push(n);
    }});
  }}
  ids.forEach((id, i) => {{ if (level[id] === undefined) level[id] = i; }});
  const lanes = {{}};
  ids.forEach(id => {{ const l = level[id]; (lanes[l]||=[]).push(id); }});
  Object.entries(lanes).forEach(([l, arr]) => {{
    arr.forEach((id, i) => {{
      pos[id] = {{ x: 80 + Number(l)*470, y: 80 + i*360 + (Number(l)%2)*80 }};
    }});
  }});
}}

function renderLegend() {{
  const legend = document.getElementById('legend');
  const blockTypes = {{'PREREQUISITE_OF':1,'FOLLOWS':1,'ELABORATES':1,'CONTRASTS':1,'PARALLEL':1}};
  legend.innerHTML = Object.entries(COLORS)
    .filter(([t]) => blockTypes[t])
    .map(([t,c]) => `<div class="legend-item"><span class="legend-line" style="background:${{c}}"></span>${{escapeHtml(edgeLabel(t))}}</div>`)
    .join('');
}}

function renderPathBtns() {{
  const container = document.getElementById('path-btns');
  if (!PATHS.length) {{
    container.innerHTML = '';
    return;
  }}
  container.innerHTML = '<span class="path-title">Paths</span>' + PATHS.map((path, i) =>
    `<button class="path-btn" id="pb${{i}}" onclick="highlightPath(${{i}})">Path ${{i+1}}</button>`
  ).join('');
}}

// --- highlight helpers ---
function applyHighlight(highlightedIds, edgeSet) {{
  // edgeSet: Set of "source->target" strings to highlight
  const allIds = new Set(BLOCKS.map(b => b.block_id));
  BLOCKS.forEach(b => {{
    const card = document.getElementById('card-' + b.block_id);
    if (!card) return;
    if (highlightedIds.has(b.block_id)) {{
      card.classList.remove('dimmed');
      card.classList.add('highlighted');
    }} else {{
      card.classList.add('dimmed');
      card.classList.remove('highlighted');
    }}
  }});
  document.querySelectorAll('#edges path').forEach(el => {{
    const key = el.dataset.src + '->' + el.dataset.tgt;
    el.style.opacity = edgeSet.has(key) ? '1' : '0.04';
  }});
  document.querySelectorAll('#labels text').forEach(el => {{
    const key = el.dataset.src + '->' + el.dataset.tgt;
    el.style.opacity = edgeSet.has(key) ? '1' : '0';
  }});
  document.getElementById('reset-btn').style.display = 'inline-block';
}}

function resetHighlight() {{
  BLOCKS.forEach(b => {{
    const card = document.getElementById('card-' + b.block_id);
    if (!card) return;
    card.classList.remove('dimmed', 'highlighted');
  }});
  document.querySelectorAll('#edges path, #labels text').forEach(el => {{
    el.style.opacity = '';
  }});
  document.querySelectorAll('.path-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('reset-btn').style.display = 'none';
  activePathIdx = null;
}}

function hoverHighlight(blockId) {{
  if (activePathIdx !== null) return; // path mode takes priority
  const connected = new Set([blockId]);
  const edgeSet = new Set();
  EDGES.forEach(e => {{
    if (e.source === blockId) {{ connected.add(e.target); edgeSet.add(e.source + '->' + e.target); }}
    if (e.target === blockId) {{ connected.add(e.source); edgeSet.add(e.source + '->' + e.target); }}
  }});
  applyHighlight(connected, edgeSet);
}}

function clearHoverHighlight() {{
  if (activePathIdx !== null) return;
  resetHighlight();
}}

function highlightPath(idx) {{
  if (activePathIdx === idx) {{ resetHighlight(); return; }}
  activePathIdx = idx;
  document.querySelectorAll('.path-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
  const path = PATHS[idx];
  const highlighted = new Set(path);
  const edgeSet = new Set();
  for (let i = 0; i < path.length - 1; i++) edgeSet.add(path[i] + '->' + path[i+1]);
  applyHighlight(highlighted, edgeSet);
}}

function microSvg(block) {{
  const micro = block.micro_graph || {{nodes:[], edges:[]}};
  const nodes = micro.nodes || [];
  const edges = micro.edges || [];
  const cx = 165, cy = 82, r = Math.min(62, 26 + nodes.length * 3);
  const nodePos = {{}};
  nodes.forEach((n, i) => {{
    const a = -Math.PI/2 + i * 2 * Math.PI / Math.max(1, nodes.length);
    nodePos[n.id] = {{ x: cx + Math.cos(a)*r, y: cy + Math.sin(a)*r }};
  }});
  const defs = Object.entries(COLORS).map(([t,c]) =>
    `<marker id="micro-arrow-${{cssId(t)}}" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5 Z" fill="${{c}}"/></marker>`
  ).join('');
  const edgeSvg = edges.map(e => {{
    const s = nodePos[e.source], t = nodePos[e.target];
    if (!s || !t) return '';
    const color = COLORS[e.type] || '#94a3b8';
    return `<line x1="${{s.x.toFixed(1)}}" y1="${{s.y.toFixed(1)}}" x2="${{t.x.toFixed(1)}}" y2="${{t.y.toFixed(1)}}" stroke="${{color}}" stroke-width="1.4" opacity=".82" marker-end="url(#micro-arrow-${{cssId(e.type)}})" />`;
  }}).join('');
  const nodeSvg = nodes.map(n => {{
    const p = nodePos[n.id];
    if (!p) return '';
    return `<g><circle class="micro-node" cx="${{p.x.toFixed(1)}}" cy="${{p.y.toFixed(1)}}" r="6"><title>${{escapeHtml(n.label||n.name||'')}}</title></circle></g>`;
  }}).join('');
  return `<svg viewBox="0 0 330 166"><defs>${{defs}}</defs>${{edgeSvg}}${{nodeSvg}}</svg>`;
}}

function renderBlocks() {{
  const layer = document.getElementById('block-layer');
  layer.innerHTML = '';
  BLOCKS.forEach(block => {{
    const p = pos[block.block_id];
    const micro = block.micro_graph || {{nodes:[], edges:[]}};
    const concepts = (block.key_concepts||[]).slice(0,6).map(c => `<span class="tag">${{texify(c)}}</span>`).join('');
    const card = document.createElement('div');
    card.className = 'block-card';
    card.id = 'card-' + block.block_id;
    card.style.left = p.x + 'px';
    card.style.top  = p.y + 'px';
    card.innerHTML = `
      <div class="head">
        <div class="block-id">${{escapeHtml(block.block_id)}} · order #${{block.order_index ?? ''}}</div>
        <div class="block-title">${{texify(block.title||block.block_id)}}</div>
      </div>
      ${{micro.nodes.length
        ? `<div class="micro">${{microSvg(block)}}</div>`
        : `<div class="micro-empty">no micro graph</div>`}}
      <div class="foot">${{concepts}}</div>`;
    installBlockDrag(card, block.block_id);
    const microEl = card.querySelector('.micro');
    if (microEl) microEl.addEventListener('click', e => {{ e.stopPropagation(); showMicroView(block); }});
    card.addEventListener('mouseenter', () => {{ showRaw(block); hoverHighlight(block.block_id); }});
    card.addEventListener('mouseleave', () => {{ if (!rawPinned) hideRaw(); clearHoverHighlight(); }});
    card.addEventListener('click', e => {{ e.stopPropagation(); rawPinned = true; showRaw(block); }});
    card.addEventListener('dblclick', e => {{ e.stopPropagation(); showMicroView(block); }});
    layer.appendChild(card);
  }});
  typeset();
}}

function showRaw(block) {{
  const panel = document.getElementById('raw-panel');
  document.getElementById('raw-title-text').innerHTML = texify(block.title||block.block_id);
  const raw = RAW_TEXTS[block.block_id] || block.summary || '';
  document.getElementById('raw-body').innerHTML = renderMarkdown(raw);
  panel.style.display = 'block';
  typeset();
}}
function hideRaw() {{ if (rawPinned) return; document.getElementById('raw-panel').style.display = 'none'; }}
function closeRaw() {{ rawPinned = false; document.getElementById('raw-panel').style.display = 'none'; }}

function renderMarkdown(s) {{
  if (!s) return '';
  const lines = s.replace(/\\r/g,'').split('\\n');
  let html = '', inList = false, tableRows = [];
  function closeList() {{ if (inList) {{ html += '</ul>'; inList = false; }} }}
  function flushTable() {{
    if (!tableRows.length) return;
    let th = '', tb = '';
    tableRows.forEach((row, ri) => {{
      const cells = row.replace(/^\\|/, '').replace(/\\|$/, '').split('|');
      if (ri === 0) th += '<tr>' + cells.map(c=>`<th style="padding:4px 10px;border:1px solid #334155;background:#1e2a3a">${{inlineMarkdown(c.trim())}}</th>`).join('') + '</tr>';
      else if (!/^[-|\\s]+$/.test(row)) tb += '<tr>' + cells.map(c=>`<td style="padding:4px 10px;border:1px solid #2f394d">${{inlineMarkdown(c.trim())}}</td>`).join('') + '</tr>';
    }});
    html += `<table style="border-collapse:collapse;margin:10px 0;width:100%"><thead>${{th}}</thead><tbody>${{tb}}</tbody></table>`;
    tableRows = [];
  }}
  for (let i = 0; i < lines.length; i++) {{
    const line = lines[i];
    if (/^\\s*$/.test(line)) {{ closeList(); flushTable(); continue; }}
    if (/^\\s*\\|/.test(line)) {{ closeList(); tableRows.push(line); continue; }}
    flushTable();
    const heading = line.match(/^\\s*(#{{1,6}})\\s+(.+)$/);
    if (heading) {{
      closeList();
      const lvl = heading[1].length;
      html += `<h${{lvl}}>${{inlineMarkdown(heading[2])}}</h${{lvl}}>`;
      continue;
    }}
    if (/^\\s*---+\\s*$/.test(line)) {{ closeList(); html += '<hr>'; continue; }}
    const li = line.match(/^\\s*[-*]\\s+(.+)$/);
    if (li) {{
      if (!inList) {{ html += '<ul>'; inList = true; }}
      html += `<li>${{inlineMarkdown(li[1])}}</li>`;
      continue;
    }}
    closeList();
    if (/^\\s{4}/.test(line) || /^```/.test(line)) {{
      const codeLines = [line.replace(/^\\s{{4}}|^```[^\\n]*/, '')];
      while (++i < lines.length && !/^```/.test(lines[i])) codeLines.push(lines[i]);
      html += `<pre>${{escapeHtml(codeLines.join('\\n'))}}</pre>`;
      continue;
    }}
    html += `<p>${{inlineMarkdown(line)}}</p>`;
  }}
  closeList(); flushTable();
  return html;
}}

function inlineMarkdown(s) {{
  return escapeHtml(String(s||''))
    .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
    .replace(/\\*(.+?)\\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code style="background:#0b1020;padding:1px 5px;border-radius:3px">$1</code>');
}}

function bigMicroSvg(block) {{
  const micro = block.micro_graph || {{nodes:[], edges:[]}};
  const nodes = micro.nodes || [], edges = micro.edges || [];
  if (!nodes.length) return '<div class="micro-empty" style="height:100%">no micro graph</div>';
  const w = 1800, h = 920, cx = w/2, cy = h/2, r = Math.min(330, 130 + nodes.length*16);
  const nodePos = {{}};
  nodes.forEach((n, i) => {{
    const a = -Math.PI/2 + i * 2*Math.PI / Math.max(1, nodes.length);
    nodePos[n.id] = {{ x: cx + Math.cos(a)*r, y: cy + Math.sin(a)*r }};
  }});
  const defs = Object.entries(COLORS).map(([t,c]) =>
    `<marker id="big-arrow-${{cssId(t)}}" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto"><path d="M0,0 L11,4 L0,8 Z" fill="${{c}}"/></marker>`
  ).join('');
  const edgeSvg = edges.map((e, i) => {{
    const s = nodePos[e.source], t = nodePos[e.target];
    if (!s || !t) return '';
    const color = COLORS[e.type] || '#94a3b8';
    const mx = (s.x+t.x)/2 + ((i%5)-2)*18, my = (s.y+t.y)/2 + ((i%3)-1)*18;
    return `<path class="big-edge" data-source="${{escapeHtml(e.source)}}" data-target="${{escapeHtml(e.target)}}" data-bx="${{((i%5)-2)*18}}" data-by="${{((i%3)-1)*18}}" d="M${{s.x}},${{s.y}} Q${{mx}},${{my}} ${{t.x}},${{t.y}}" fill="none" stroke="${{color}}" stroke-width="2.5" opacity=".9" marker-end="url(#big-arrow-${{cssId(e.type)}})" /><text class="edge-label big-edge-label" data-source="${{escapeHtml(e.source)}}" data-target="${{escapeHtml(e.target)}}" data-bx="${{((i%5)-2)*18}}" data-by="${{((i%3)-1)*18}}" x="${{mx}}" y="${{my-8}}" text-anchor="middle" fill="${{color}}">${{escapeHtml(edgeLabel(e.type))}}</text>`;
  }}).join('');
  const nodeSvg = nodes.map(n => {{
    const p = nodePos[n.id];
    if (!p) return '';
    return `<g class="big-node-group" data-node-id="${{escapeHtml(n.id)}}" data-x="${{p.x}}" data-y="${{p.y}}" transform="translate(${{p.x}},${{p.y}})"><circle class="big-node" cx="0" cy="0" r="12"/><foreignObject x="-115" y="-62" width="230" height="54"><div xmlns="http://www.w3.org/1999/xhtml" class="big-node-label">${{texify(trunc(n.label||n.name,64))}}</div></foreignObject></g>`;
  }}).join('');
  return `<svg viewBox="0 0 ${{w}} ${{h}}"><defs>${{defs}}</defs>${{edgeSvg}}${{nodeSvg}}</svg>`;
}}

function bigPos(id) {{
  const g = document.querySelector(`.big-node-group[data-node-id="${{CSS.escape(id)}}"]`);
  return g ? {{ x: Number(g.dataset.x), y: Number(g.dataset.y) }} : null;
}}
function updateBigEdges() {{
  document.querySelectorAll('.big-edge').forEach(path => {{
    const s = bigPos(path.dataset.source), t = bigPos(path.dataset.target);
    if (!s||!t) return;
    const bx = Number(path.dataset.bx||0), by = Number(path.dataset.by||0);
    const mx = (s.x+t.x)/2+bx, my = (s.y+t.y)/2+by;
    path.setAttribute('d', `M${{s.x}},${{s.y}} Q${{mx}},${{my}} ${{t.x}},${{t.y}}`);
  }});
  document.querySelectorAll('.big-edge-label').forEach(label => {{
    const s = bigPos(label.dataset.source), t = bigPos(label.dataset.target);
    if (!s||!t) return;
    label.setAttribute('x', (s.x+t.x)/2 + Number(label.dataset.bx||0));
    label.setAttribute('y', (s.y+t.y)/2 + Number(label.dataset.by||0) - 8);
  }});
}}
function installBigMicroDrag() {{
  const svg = document.querySelector('#micro-canvas svg');
  if (!svg) return;
  let active = null, start = null;
  svg.querySelectorAll('.big-node-group').forEach(g => {{
    g.addEventListener('mousedown', e => {{
      e.stopPropagation();
      active = g;
      const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      start = {{ x: loc.x, y: loc.y, ox: Number(g.dataset.x), oy: Number(g.dataset.y) }};
    }});
  }});
  window.addEventListener('mousemove', e => {{
    if (!active||!start) return;
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    const nx = start.ox + loc.x - start.x, ny = start.oy + loc.y - start.y;
    active.dataset.x = nx; active.dataset.y = ny;
    active.setAttribute('transform', `translate(${{nx}},${{ny}})`);
    updateBigEdges();
  }});
  window.addEventListener('mouseup', () => {{ active = null; start = null; }});
}}

function showMicroView(block) {{
  const view = document.getElementById('micro-view');
  document.getElementById('micro-title').innerHTML = texify(block.title||block.block_id);
  document.getElementById('micro-canvas').innerHTML = bigMicroSvg(block);
  view.style.display = 'block';
  hideRaw();
  installBigMicroDrag();
  typeset();
}}
function hideMicroView() {{
  document.getElementById('micro-view').style.display = 'none';
  document.getElementById('micro-canvas').innerHTML = '';
}}

function renderEdges() {{
  const g = document.getElementById('edges'), labels = document.getElementById('labels'), defs = document.getElementById('defs');
  defs.innerHTML = Object.entries(COLORS).map(([t,c]) =>
    `<marker id="arrow-${{cssId(t)}}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="${{c}}"/></marker>`
  ).join('');
  g.innerHTML = ''; labels.innerHTML = '';
  EDGES.forEach((e, i) => {{
    const s = pos[e.source], t = pos[e.target];
    if (!s||!t) return;
    const sx = s.x+BLOCK_W, sy = s.y+BLOCK_H/2;
    const tx = t.x,         ty = t.y+BLOCK_H/2;
    const bend = ((i%3)-1)*42;
    const mx = (sx+tx)/2, my = (sy+ty)/2 + bend;
    const d = `M${{sx}},${{sy}} Q${{mx}},${{my}} ${{tx}},${{ty}}`;
    const color = COLORS[e.type] || '#94a3b8';
    g.insertAdjacentHTML('beforeend',
      `<path data-src="${{escapeHtml(e.source)}}" data-tgt="${{escapeHtml(e.target)}}" d="${{d}}" fill="none" stroke="${{color}}" stroke-width="2.8" opacity=".92" marker-end="url(#arrow-${{cssId(e.type)}})" />`);
    labels.insertAdjacentHTML('beforeend',
      `<text data-src="${{escapeHtml(e.source)}}" data-tgt="${{escapeHtml(e.target)}}" class="edge-label" x="${{mx}}" y="${{my-8}}" text-anchor="middle" fill="${{color}}">${{escapeHtml(edgeLabel(e.type))}}</text>`);
  }});
}}

function installBlockDrag(card, id) {{
  let dragging = false, sx = 0, sy = 0, ox = 0, oy = 0;
  card.addEventListener('mousedown', e => {{
    e.stopPropagation();
    dragging = true; sx = e.clientX; sy = e.clientY; ox = pos[id].x; oy = pos[id].y;
    card.classList.add('focused');
  }});
  window.addEventListener('mousemove', e => {{
    if (!dragging) return;
    pos[id].x = ox + (e.clientX-sx)/scale;
    pos[id].y = oy + (e.clientY-sy)/scale;
    card.style.left = pos[id].x + 'px';
    card.style.top  = pos[id].y + 'px';
    renderEdges();
  }});
  window.addEventListener('mouseup', () => {{ dragging = false; card.classList.remove('focused'); }});
}}

function applyWorld() {{
  document.getElementById('world').style.transform = `translate(${{panX}}px,${{panY}}px) scale(${{scale}})`;
}}

const stage = document.getElementById('stage');
stage.addEventListener('mousedown', e => {{
  if (e.target.closest('.block-card')) return;
  panning = true; panStartX = e.clientX - panX; panStartY = e.clientY - panY;
}});
window.addEventListener('mousemove', e => {{
  if (panning) {{ panX = e.clientX - panStartX; panY = e.clientY - panStartY; applyWorld(); }}
}});
window.addEventListener('mouseup', () => {{ panning = false; }});
stage.addEventListener('click', e => {{
  if (e.target.closest('.block-card') || e.target.closest('#raw-panel')) return;
  closeRaw();
}});
stage.addEventListener('wheel', e => {{
  e.preventDefault();
  const delta = e.deltaY < 0 ? 1.08 : 0.925;
  const rect = stage.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  panX = mx - (mx-panX)*delta;
  panY = my - (my-panY)*delta;
  scale = Math.max(.25, Math.min(2.4, scale*delta));
  applyWorld();
}}, {{passive:false}});

layout();
renderLegend();
renderPathBtns();
renderBlocks();
renderEdges();
applyWorld();
document.getElementById('back-btn').addEventListener('click', hideMicroView);
document.getElementById('raw-close').addEventListener('click', e => {{ e.stopPropagation(); closeRaw(); }});
window.addEventListener('load', () => setTimeout(typeset, 150));
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    src = Path(args.json_path)
    if not src.exists():
        print(f"File not found: {src}", file=sys.stderr)
        sys.exit(1)

    data      = json.loads(src.read_text(encoding="utf-8"))
    blocks    = data.get("block_graph", {}).get("blocks", [])
    raw_texts = load_raw_texts(src, blocks)

    html     = build_html(data, raw_texts, src)
    out_path = src.with_name("block_graph_viz.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path}")

    if not args.no_open:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
