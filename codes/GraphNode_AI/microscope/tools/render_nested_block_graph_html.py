#!/usr/bin/env python3
"""Render a draggable block graph with nested per-block micro graphs."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


EDGE_COLORS = {
    "PREREQUISITE_OF": "#ef4444",
    "FOLLOWS": "#38bdf8",
    "ELABORATES": "#22c55e",
    "CONTRASTS": "#f59e0b",
    "PARALLEL": "#94a3b8",
    "defines": "#a78bfa",
    "uses": "#14b8a6",
    "requires": "#ef4444",
    "causes": "#f97316",
    "derives_from": "#60a5fa",
    "equivalent_to": "#c084fc",
    "part_of": "#2dd4bf",
    "applied_in": "#38bdf8",
    "influences": "#f472b6",
    "leads_to": "#fb7185",
    "example_of": "#fbbf24",
}

FALLBACK_COLORS = ["#60a5fa", "#34d399", "#fbbf24", "#fb7185", "#a78bfa", "#2dd4bf"]


def safe_json(value) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def block_id(block: dict) -> str:
    return block.get("block_id") or block.get("id") or ""


def edge_type_colors(blocks: list[dict], edges: list[dict]) -> dict[str, str]:
    types = {e.get("type", "RELATED") for e in edges}
    for block in blocks:
        for edge in (block.get("micro_graph") or {}).get("edges", []):
            types.add(edge.get("type", "related_to"))
    colors = {}
    fallback_i = 0
    for typ in sorted(types):
        if typ in EDGE_COLORS:
            colors[typ] = EDGE_COLORS[typ]
        else:
            colors[typ] = FALLBACK_COLORS[fallback_i % len(FALLBACK_COLORS)]
            fallback_i += 1
    return colors


def normalize_blocks(raw_blocks: list[dict]) -> list[dict]:
    out = []
    for block in sorted(raw_blocks, key=lambda b: b.get("order_index", 999)):
        b = dict(block)
        b["block_id"] = block_id(b)
        micro = b.get("micro_graph") or {"nodes": [], "edges": []}
        nodes = []
        name_to_id = {}
        for i, node in enumerate(micro.get("nodes", [])):
            n = dict(node)
            nid = n.get("id") or f"{b['block_id']}::n{i}"
            n["id"] = nid
            n["label"] = n.get("label") or n.get("name") or nid
            nodes.append(n)
            for key in (n.get("id"), n.get("name"), n.get("label")):
                if key:
                    name_to_id[key] = nid
        edges = []
        for i, edge in enumerate(micro.get("edges", [])):
            e = dict(edge)
            source = e.get("source") or e.get("start")
            target = e.get("target")
            e["id"] = e.get("id") or f"{b['block_id']}::e{i}"
            e["source"] = name_to_id.get(source, source)
            e["target"] = name_to_id.get(target, target)
            e["type"] = e.get("type") or "related_to"
            edges.append(e)
        b["micro_graph"] = {"nodes": nodes, "edges": edges}
        out.append(b)
    return out


def load_raw_texts(src: Path, blocks: list[dict]) -> dict[str, str]:
    micro_dir = src.parent / "03_micro_graphs"
    raw_texts = {}
    for block in blocks:
        bid = block_id(block)
        raw_path = micro_dir / bid / "raw_text.txt"
        raw_texts[bid] = raw_path.read_text(encoding="utf-8") if raw_path.exists() else ""
    return raw_texts


def build_html(data: dict, src: Path) -> str:
    bg = data.get("block_graph", {})
    blocks = normalize_blocks(bg.get("nodes") or bg.get("blocks") or [])
    edges = bg.get("edges", [])
    paths = bg.get("paths", [])
    colors = edge_type_colors(blocks, edges)
    raw_texts = load_raw_texts(src, blocks)
    source_name = data.get("source_name") or src.name
    total_micro_nodes = sum(len((b.get("micro_graph") or {}).get("nodes", [])) for b in blocks)
    total_micro_edges = sum(len((b.get("micro_graph") or {}).get("edges", [])) for b in blocks)

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(source_name)} - block graph</title>
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
#edge-svg {{ position:absolute; inset:0; width:2400px; height:1200px; pointer-events:none; overflow:visible; }}
#block-layer {{ position:absolute; inset:0; }}
.block-card {{ position:absolute; width:330px; height:286px; background:#202738; border:1px solid #55627a; border-radius:8px; overflow:hidden; box-shadow:0 18px 38px #0008; cursor:grab; }}
.block-card:active {{ cursor:grabbing; }}
.block-card.focused {{ border-color:#cbd5e1; box-shadow:0 0 0 2px #e5e7eb33, 0 18px 42px #000a; }}
.head {{ height:78px; padding:14px 16px 10px; background:#151b28; user-select:none; }}
.block-id {{ font:12px Consolas,monospace; color:#7d8798; margin-bottom:6px; }}
.block-title {{ color:#f8fafc; font-weight:800; font-size:17px; line-height:1.25; max-height:42px; overflow:hidden; }}
.micro {{ position:relative; height:166px; background:#252d3f; border-top:1px solid #2f394d; border-bottom:1px solid #2f394d; }}
.micro svg {{ width:100%; height:100%; display:block; }}
.micro-empty {{ height:166px; display:flex; align-items:center; justify-content:center; color:#7d8798; font-size:12px; background:#252d3f; }}
.micro-node-label {{ width:86px; max-height:26px; color:#e5e7eb; font-size:9px; line-height:1.12; text-align:center; overflow:hidden; }}
.micro-node-label mjx-container {{ display:inline-block !important; max-width:82px !important; max-height:24px !important; overflow:hidden !important; font-size:80% !important; }}
.micro-node-label mjx-container svg {{ max-width:82px !important; max-height:22px !important; }}
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
.big-node-label mjx-container {{ display:inline-block !important; max-width:220px !important; max-height:48px !important; overflow:hidden !important; font-size:78% !important; }}
.big-node-label mjx-container svg {{ max-width:220px !important; max-height:44px !important; }}
.big-node-group {{ cursor:grab; }}
.big-node-group:active {{ cursor:grabbing; }}
</style>
</head>
<body>
<div id="topbar">
  <div id="title">{html.escape(source_name)}</div>
  <div id="meta">{len(blocks)} blocks · {len(edges)} block edges · {total_micro_nodes} micro nodes · {total_micro_edges} micro edges</div>
  <div id="legend"></div>
</div>
<div id="stage">
  <div id="world">
    <svg id="edge-svg"><defs id="defs"></defs><g id="edges"></g><g id="labels"></g></svg>
    <div id="block-layer"></div>
  </div>
  <div class="hint">Drag block headers to move blocks. Wheel to zoom, drag empty space to pan.</div>
  <div id="raw-panel"><div id="raw-title"><button id="raw-close">x</button><span id="raw-title-text"></span></div><div id="raw-body"></div></div>
  <div id="micro-view">
    <div id="micro-toolbar"><button id="back-btn">Back</button><div id="micro-title"></div></div>
    <div id="micro-canvas"></div>
  </div>
</div>
<script>
const BLOCKS = {safe_json(blocks)};
const EDGES = {safe_json(edges)};
const PATHS = {safe_json(paths)};
const COLORS = {safe_json(colors)};
const RAW_TEXTS = {safe_json(raw_texts)};
const KO = {{
  PREREQUISITE_OF: 'text',
  FOLLOWS: 'text',
  ELABORATES: 'text',
  CONTRASTS: 'text',
  PARALLEL: 'text',
  RELATED: 'text',
  defines: 'text',
  uses: 'text',
  part_of: 'text',
  causes: 'text',
  influences: 'text',
  leads_to: 'text',
  solves: 'text',
  improves: 'text',
  requires: 'text',
  prerequisite_of: 'text',
  derives_from: 'text',
  equivalent_to: 'text',
  example_of: 'text',
  contrasts_with: 'text',
  applied_in: 'text',
  occurs_in: 'text',
  happens_during: 'processing text',
  proposed_by: 'text',
  related_to: 'text',
  has_property: 'text',
  has_example: 'text',
  supports: 'text',
  explains: 'text',
  contains: 'text'
}};
const BLOCK_W = 330, BLOCK_H = 286;
let pos = {{}};
let panX = 20, panY = 20, scale = 1;
let panning = false, panStartX = 0, panStartY = 0;
let rawPinned = false;

function texify(s) {{
  s = String(s || '');
  const escaped = escapeHtml(s);
  if (s.includes('\\\\(') || s.includes('\\\\[')) return escaped;
  const compactMath = /\\\\(frac|int|sum|prod|sin|cos|tan|sec|mathbb|left|right)|[_^=]/.test(s) && !/[\uAC00-\uD7A3A-Za-z]{{2,}}\\s+[\uAC00-\uD7A3A-Za-z]{{2,}}/.test(s);
  if (compactMath) return '\\\\(' + escaped + '\\\\)';
  return escaped
    .replace(/\\b(sin|cos|tan|sec)\\s*\\^\\s*([0-9]+)\\s*([a-zA-Z])\\b/g, (_, f, pow, v) => `\\\\(${{f}}^{{${{pow}}}} ${{v}}\\\\)`)
    .replace(/\\b(sin|cos|tan|sec)\\s+([a-zA-Z])\\b/g, (_, f, v) => `\\\\(${{f}} ${{v}}\\\\)`)
    .replace(/\\bpi\\b/g, '\\\\(\\\\pi\\\\)');
}}
function escapeHtml(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}
function edgeLabel(t) {{ return KO[t] || t || ''; }}
function trunc(s, n) {{ s = String(s || ''); return s.length > n ? s.slice(0, n - 3) + '...' : s; }}
function typeset() {{
  if (window.MathJax?.typesetPromise) {{
    MathJax.typesetPromise().catch(() => {{}});
  }}
}}

function layout() {{
  const ids = BLOCKS.map(b => b.block_id);
  const indeg = Object.fromEntries(ids.map(id => [id, 0]));
  const adj = Object.fromEntries(ids.map(id => [id, []]));
  EDGES.forEach(e => {{ if (adj[e.source]) {{ adj[e.source].push(e.target); indeg[e.target] = (indeg[e.target] || 0) + 1; }} }});
  const q = ids.filter(id => !indeg[id]);
  const level = Object.fromEntries(q.map(id => [id, 0]));
  for (let qi = 0; qi < q.length; qi++) {{
    const cur = q[qi];
    (adj[cur] || []).forEach(n => {{ level[n] = Math.max(level[n] || 0, (level[cur] || 0) + 1); if (--indeg[n] === 0) q.push(n); }});
  }}
  ids.forEach((id, i) => {{ if (level[id] === undefined) level[id] = i; }});
  const lanes = {{}};
  ids.forEach(id => {{ const l = level[id]; (lanes[l] ||= []).push(id); }});
  Object.entries(lanes).forEach(([l, arr]) => {{
    arr.forEach((id, i) => {{ pos[id] = {{ x:80 + Number(l) * 470, y:80 + i * 360 + (Number(l) % 2) * 80 }}; }});
  }});
}}

function renderLegend() {{
  const legend = document.getElementById('legend');
  legend.innerHTML = Object.entries(COLORS).map(([t, c]) =>
    `<div class="legend-item"><span class="legend-line" style="background:${{c}}"></span>${{escapeHtml(edgeLabel(t))}}</div>`
  ).join('');
}}

function renderBlocks() {{
  const layer = document.getElementById('block-layer');
  layer.innerHTML = '';
  BLOCKS.forEach(block => {{
    const p = pos[block.block_id];
    const micro = block.micro_graph || {{nodes:[], edges:[]}};
    const concepts = (block.key_concepts || []).slice(0, 5).map(c => `<span class="tag">${{texify(c)}}</span>`).join('');
    const card = document.createElement('div');
    card.className = 'block-card';
    card.id = 'card-' + block.block_id;
    card.style.left = p.x + 'px';
    card.style.top = p.y + 'px';
    card.innerHTML = `
      <div class="head">
        <div class="block-id">${{escapeHtml(block.block_id)}} · order #${{block.order_index ?? ''}}</div>
        <div class="block-title">${{texify(block.title || block.block_id)}}</div>
      </div>
      ${{micro.nodes.length ? `<div class="micro">${{microSvg(block)}}</div>` : `<div class="micro-empty">no micro graph</div>`}}
      <div class="foot">${{concepts}}</div>`;
    installBlockDrag(card, block.block_id);
    const microEl = card.querySelector('.micro');
    if (microEl) microEl.addEventListener('click', e => {{
      e.stopPropagation();
      showMicroView(block);
    }});
    card.addEventListener('mouseenter', () => showRaw(block));
    card.addEventListener('mouseleave', () => {{ if (!rawPinned) hideRaw(); }});
    card.addEventListener('click', e => {{
      e.stopPropagation();
      rawPinned = true;
      showRaw(block);
    }});
    card.addEventListener('dblclick', e => {{
      e.stopPropagation();
      showMicroView(block);
    }});
    layer.appendChild(card);
  }});
  typeset();
}}

function showRaw(block) {{
  const panel = document.getElementById('raw-panel');
  const title = document.getElementById('raw-title-text');
  const body = document.getElementById('raw-body');
  title.innerHTML = texify(block.title || block.block_id);
  const raw = RAW_TEXTS[block.block_id] || block.summary || '';
  body.innerHTML = renderMarkdown(raw);
  panel.style.display = 'block';
  typeset();
}}

function hideRaw() {{
  if (rawPinned) return;
  document.getElementById('raw-panel').style.display = 'none';
}}

function closeRaw() {{
  rawPinned = false;
  document.getElementById('raw-panel').style.display = 'none';
}}

function texifyMultiline(s) {{
  return escapeHtml(String(s || ''))
    .replace(/\\\\\[((?:.|\\n)*?)\\\\\]/g, (_, m) => '\\\\[' + m + '\\\\]')
    .replace(/\\\\\(((?:.|\\n)*?)\\\\\)/g, (_, m) => '\\\\(' + m + '\\\\)');
}}

function restoreTex(s) {{
  return s
    .replace(/\\\\\[((?:.|\\n)*?)\\\\\]/g, (_, m) => '\\\\[' + m + '\\\\]')
    .replace(/\\\\\(((?:.|\\n)*?)\\\\\)/g, (_, m) => '\\\\(' + m + '\\\\)');
}}

function inlineMarkdown(s) {{
  return restoreTex(escapeHtml(s))
    .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}}

function renderMarkdown(s) {{
  const lines = String(s || '').replace(/\\r\\n/g, '\\n').split('\\n');
  let html = '', inList = false, inTable = false, tableRows = [];
  function closeList() {{ if (inList) {{ html += '</ul>'; inList = false; }} }}
  function flushTable() {{
    if (!inTable) return;
    html += '<pre>' + tableRows.map(escapeHtml).join('\\n') + '</pre>';
    tableRows = []; inTable = false;
  }}
  for (let i = 0; i < lines.length; i++) {{
    const line = lines[i];
    if (/^\s*$/.test(line)) {{ closeList(); flushTable(); continue; }}
    if (/^\s*\|/.test(line)) {{ closeList(); inTable = true; tableRows.push(line); continue; }}
    flushTable();
    if (line.trim() === '\\\\[') {{
      closeList();
      const mathLines = [];
      while (++i < lines.length && lines[i].trim() !== '\\\\]') {{
        mathLines.push(lines[i]);
      }}
      html += `<div class="math-block">\\\\[${{escapeHtml(mathLines.join('\\n'))}}\\\\]</div>`;
      continue;
    }}
    const heading = line.match(/^\s*(#{{1,6}})\s+(.+)$/);
    if (heading) {{
      closeList();
      const level = Math.min(4, heading[1].length);
      html += `<h${{level}}>${{inlineMarkdown(heading[2])}}</h${{level}}>`;
      continue;
    }}
    if (/^\s*---+\s*$/.test(line)) {{ closeList(); html += '<hr>'; continue; }}
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) {{
      if (!inList) {{ html += '<ul>'; inList = true; }}
      html += `<li>${{inlineMarkdown(bullet[1])}}</li>`;
      continue;
    }}
    closeList();
    html += `<p>${{inlineMarkdown(line)}}</p>`;
  }}
  closeList(); flushTable();
  return html;
}}

function showMicroView(block) {{
  const view = document.getElementById('micro-view');
  document.getElementById('micro-title').innerHTML = texify(block.title || block.block_id);
  document.getElementById('micro-canvas').innerHTML = bigMicroSvg(block);
  view.style.display = 'block';
  hideRaw();
  installBigMicroDrag();
  typeset();
}}

function hideMicroView() {{
  document.getElementById('micro-view').style.display = 'none';
  document.getElementById('micro-canvas').innerHTML = '';
  typeset();
}}

function bigMicroSvg(block) {{
  const micro = block.micro_graph || {{nodes:[], edges:[]}};
  const nodes = micro.nodes || [];
  const edges = micro.edges || [];
  if (!nodes.length) return '<div class="micro-empty" style="height:100%">no micro graph</div>';
  const w = 1800, h = 920, cx = w / 2, cy = h / 2, r = Math.min(330, 130 + nodes.length * 16);
  const nodePos = {{}};
  nodes.forEach((n, i) => {{
    const a = -Math.PI / 2 + i * 2 * Math.PI / Math.max(1, nodes.length);
    nodePos[n.id] = {{ x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r }};
  }});
  const defs = Object.entries(COLORS).map(([t,c]) => `<marker id="big-arrow-${{cssId(t)}}" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto"><path d="M0,0 L11,4 L0,8 Z" fill="${{c}}"/></marker>`).join('');
  const edgeSvg = edges.map((e, i) => {{
    const s = nodePos[e.source], t = nodePos[e.target];
    if (!s || !t) return '';
    const color = COLORS[e.type] || '#94a3b8';
    const mx = (s.x + t.x) / 2 + ((i % 5) - 2) * 18;
    const my = (s.y + t.y) / 2 + ((i % 3) - 1) * 18;
    return `<path class="big-edge" data-source="${{escapeHtml(e.source)}}" data-target="${{escapeHtml(e.target)}}" data-bx="${{((i % 5) - 2) * 18}}" data-by="${{((i % 3) - 1) * 18}}" d="M${{s.x}},${{s.y}} Q${{mx}},${{my}} ${{t.x}},${{t.y}}" fill="none" stroke="${{color}}" stroke-width="2.5" opacity=".9" marker-end="url(#big-arrow-${{cssId(e.type)}})" /><text class="edge-label big-edge-label" data-source="${{escapeHtml(e.source)}}" data-target="${{escapeHtml(e.target)}}" data-bx="${{((i % 5) - 2) * 18}}" data-by="${{((i % 3) - 1) * 18}}" x="${{mx}}" y="${{my - 8}}" text-anchor="middle" fill="${{color}}">${{escapeHtml(edgeLabel(e.type))}}</text>`;
  }}).join('');
  const nodeSvg = nodes.map(n => {{
    const p = nodePos[n.id];
    return `<g class="big-node-group" data-node-id="${{escapeHtml(n.id)}}" data-x="${{p.x}}" data-y="${{p.y}}" transform="translate(${{p.x}},${{p.y}})"><circle class="big-node" cx="0" cy="0" r="12" /><foreignObject x="-115" y="-62" width="230" height="54"><div xmlns="http://www.w3.org/1999/xhtml" class="big-node-label">${{texify(trunc(n.label || n.name, 64))}}</div></foreignObject></g>`;
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
    if (!s || !t) return;
    const bx = Number(path.dataset.bx || 0), by = Number(path.dataset.by || 0);
    const mx = (s.x + t.x) / 2 + bx, my = (s.y + t.y) / 2 + by;
    path.setAttribute('d', `M${{s.x}},${{s.y}} Q${{mx}},${{my}} ${{t.x}},${{t.y}}`);
  }});
  document.querySelectorAll('.big-edge-label').forEach(label => {{
    const s = bigPos(label.dataset.source), t = bigPos(label.dataset.target);
    if (!s || !t) return;
    const bx = Number(label.dataset.bx || 0), by = Number(label.dataset.by || 0);
    label.setAttribute('x', (s.x + t.x) / 2 + bx);
    label.setAttribute('y', (s.y + t.y) / 2 + by - 8);
  }});
}}

function installBigMicroDrag() {{
  const svg = document.querySelector('#micro-canvas svg');
  if (!svg) return;
  let active = null;
  let start = null;
  svg.querySelectorAll('.big-node-group').forEach(g => {{
    g.addEventListener('mousedown', e => {{
      e.stopPropagation();
      active = g;
      const pt = svg.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      start = {{ x: loc.x, y: loc.y, ox: Number(g.dataset.x), oy: Number(g.dataset.y) }};
    }});
  }});
  window.addEventListener('mousemove', e => {{
    if (!active || !start) return;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const loc = pt.matrixTransform(svg.getScreenCTM().inverse());
    const nx = start.ox + loc.x - start.x;
    const ny = start.oy + loc.y - start.y;
    active.dataset.x = nx;
    active.dataset.y = ny;
    active.setAttribute('transform', `translate(${{nx}},${{ny}})`);
    updateBigEdges();
  }});
  window.addEventListener('mouseup', () => {{ active = null; start = null; }});
}}

function microSvg(block) {{
  const micro = block.micro_graph || {{nodes:[], edges:[]}};
  const nodes = micro.nodes || [];
  const edges = micro.edges || [];
  const cx = 165, cy = 82, r = Math.min(62, 26 + nodes.length * 3);
  const nodePos = {{}};
  nodes.forEach((n, i) => {{
    const a = -Math.PI / 2 + i * 2 * Math.PI / Math.max(1, nodes.length);
    nodePos[n.id] = {{ x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r }};
  }});
  const edgeSvg = edges.map(e => {{
    const s = nodePos[e.source], t = nodePos[e.target];
    if (!s || !t) return '';
    const color = COLORS[e.type] || '#94a3b8';
    return `<line x1="${{s.x.toFixed(1)}}" y1="${{s.y.toFixed(1)}}" x2="${{t.x.toFixed(1)}}" y2="${{t.y.toFixed(1)}}" stroke="${{color}}" stroke-width="1.4" opacity=".82" marker-end="url(#micro-arrow-${{cssId(e.type)}})" />`;
  }}).join('');
  const nodeSvg = nodes.map(n => {{
    const p = nodePos[n.id];
    return `<g><circle class="micro-node" cx="${{p.x.toFixed(1)}}" cy="${{p.y.toFixed(1)}}" r="6"><title>${{escapeHtml(n.label || n.name || '')}}</title></circle></g>`;
  }}).join('');
  const defs = Object.entries(COLORS).map(([t,c]) => `<marker id="micro-arrow-${{cssId(t)}}" markerWidth="7" markerHeight="5" refX="6" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5 Z" fill="${{c}}"/></marker>`).join('');
  return `<svg viewBox="0 0 330 166"><defs>${{defs}}</defs>${{edgeSvg}}${{nodeSvg}}</svg>`;
}}

function cssId(s) {{ return String(s || 'x').replace(/[^a-zA-Z0-9_-]/g, '_'); }}

function renderEdges() {{
  const g = document.getElementById('edges'), labels = document.getElementById('labels'), defs = document.getElementById('defs');
  defs.innerHTML = Object.entries(COLORS).map(([t,c]) => `<marker id="arrow-${{cssId(t)}}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="${{c}}"/></marker>`).join('');
  g.innerHTML = ''; labels.innerHTML = '';
  EDGES.forEach((e, i) => {{
    const s = pos[e.source], t = pos[e.target];
    if (!s || !t) return;
    const sx = s.x + BLOCK_W, sy = s.y + BLOCK_H / 2;
    const tx = t.x, ty = t.y + BLOCK_H / 2;
    const bend = ((i % 3) - 1) * 42;
    const mx = (sx + tx) / 2, my = (sy + ty) / 2 + bend;
    const d = `M${{sx}},${{sy}} Q${{mx}},${{my}} ${{tx}},${{ty}}`;
    const color = COLORS[e.type] || '#94a3b8';
    g.insertAdjacentHTML('beforeend', `<path d="${{d}}" fill="none" stroke="${{color}}" stroke-width="2.8" opacity=".92" marker-end="url(#arrow-${{cssId(e.type)}})" />`);
    labels.insertAdjacentHTML('beforeend', `<text class="edge-label" x="${{mx}}" y="${{my - 8}}" text-anchor="middle" fill="${{color}}">${{escapeHtml(edgeLabel(e.type))}}</text>`);
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
    pos[id].x = ox + (e.clientX - sx) / scale;
    pos[id].y = oy + (e.clientY - sy) / scale;
    card.style.left = pos[id].x + 'px';
    card.style.top = pos[id].y + 'px';
    renderEdges();
  }});
  window.addEventListener('mouseup', () => {{ dragging = false; card.classList.remove('focused'); }});
}}

function applyWorld() {{
  document.getElementById('world').style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
}}
const stage = document.getElementById('stage');
stage.addEventListener('mousedown', e => {{
  if (e.target.closest('.block-card')) return;
  panning = true; panStartX = e.clientX - panX; panStartY = e.clientY - panY;
}});
window.addEventListener('mousemove', e => {{ if (panning) {{ panX = e.clientX - panStartX; panY = e.clientY - panStartY; applyWorld(); }} }});
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
  panX = mx - (mx - panX) * delta;
  panY = my - (my - panY) * delta;
  scale = Math.max(.25, Math.min(2.4, scale * delta));
  applyWorld();
}}, {{ passive:false }});

layout();
renderLegend();
renderBlocks();
renderEdges();
applyWorld();
document.getElementById('back-btn').addEventListener('click', hideMicroView);
document.getElementById('raw-close').addEventListener('click', e => {{ e.stopPropagation(); closeRaw(); }});
window.addEventListener('load', () => setTimeout(typeset, 150));
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--out")
    args = parser.parse_args()

    src = Path(args.json_path)
    data = json.loads(src.read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else src.with_name("block_graph_viz.html")
    out.write_text(build_html(data, src), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
