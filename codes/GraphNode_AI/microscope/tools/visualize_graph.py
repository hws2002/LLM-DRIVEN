"""English documentation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import urllib.request
from pathlib import Path

VIS_CDN = "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"

def _get_vis_js() -> str:
    try:
        with urllib.request.urlopen(VIS_CDN, timeout=10) as r:
            return r.read().decode("utf-8")
    except Exception:
        return None

# English comment.
TYPE_COLORS: dict[str, str] = {
    "Concept":   "#4E79A7",
    "Algorithm": "#F28E2B",
    "Structure": "#E15759",
    "Formula":   "#76B7B2",
    "Theorem":   "#59A14F",
    "Principle": "#EDC948",
    "Problem":   "#B07AA1",
    "Solution":  "#FF9DA7",
    "Example":   "#9C755F",
    "Domain":    "#BAB0AC",
    "Person":    "#D37295",
    "Tool":      "#A0CBE8",
}
DEFAULT_COLOR = "#CCCCCC"


def _find_standardized(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        candidates = sorted(p.glob("*_standardized.json")) + sorted(p.glob("standardized.json"))
        if not candidates:
            raise FileNotFoundError(f"standardized.json not found in {p}")
        return candidates[-1]
    raise FileNotFoundError(f"Path not found: {path}")


def _type_initial_positions(types_seen: list[str], radius: float = 600) -> dict[str, tuple[float, float]]:
    """English documentation."""
    positions = {}
    n = len(types_seen)
    for i, t in enumerate(types_seen):
        angle = 2 * math.pi * i / n
        positions[t] = (radius * math.cos(angle), radius * math.sin(angle))
    return positions


def build_graph(standardized: list[dict]) -> tuple[list[dict], list[dict]]:
    # key: (name, type) → node
    nodes_map: dict[tuple[str, str], dict] = {}
    # English comment.
    name_to_ids: dict[str, list[str]] = {}
    raw_edges: list[dict] = []
    # English comment.
    types_order: list[str] = []
    types_seen: set[str] = set()

    for batch in standardized:
        for node in batch.get("nodes", []):
            name = node.get("name", "")
            if not name:
                continue
            ntype = node.get("type", "")
            types = ntype if isinstance(ntype, list) else ([ntype] if ntype else ["Unknown"])
            desc = node.get("description", "")

            for t in types:
                if t not in types_seen:
                    types_seen.add(t)
                    types_order.append(t)

                key = (name, t)
                if key not in nodes_map:
                    node_id = f"{name}||{t}"
                    nodes_map[key] = {
                        "id": node_id,
                        "label": name,
                        "title": f"<b>{name}</b><br>Type: {t}<br>{desc}",
                        "color": TYPE_COLORS.get(t, DEFAULT_COLOR),
                        "_type": t,
                    }
                    name_to_ids.setdefault(name, []).append(node_id)

        for edge in batch.get("edges", []):
            raw_edges.append(edge)

    # English comment.
    edges: list[dict] = []
    seen_edges: set[tuple] = set()
    for edge in raw_edges:
        start = edge.get("start", "")
        target = edge.get("target", "")
        etype = edge.get("type", "")
        if not start or not target or not etype:
            continue
        desc = edge.get("description", "")
        title = f"<b>{etype}</b><br>{desc}" if desc else etype

        from_ids = name_to_ids.get(start, [start])
        to_ids = name_to_ids.get(target, [target])
        for fid in from_ids:
            for tid in to_ids:
                key = (fid, tid, etype)
                if key not in seen_edges:
                    seen_edges.add(key)
                    edges.append({"from": fid, "to": tid, "label": etype, "title": title, "arrows": "to"})

    # English comment.
    rng = random.Random(42)
    type_pos = _type_initial_positions(types_order)
    jitter = 80
    nodes_list = []
    for node in nodes_map.values():
        t = node.pop("_type")
        cx, cy = type_pos.get(t, (0, 0))
        node["x"] = cx + rng.uniform(-jitter, jitter)
        node["y"] = cy + rng.uniform(-jitter, jitter)
        nodes_list.append(node)

    return nodes_list, edges


def render_html(nodes: list[dict], edges: list[dict], title: str) -> str:
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)
    vis_js = _get_vis_js()
    vis_script = f"<script>{vis_js}</script>" if vis_js else f'<script src="{VIS_CDN}"></script>'

    type_legend = "\n".join(
        f'<span style="background:{color};padding:2px 8px;border-radius:3px;margin:2px;display:inline-block;font-size:12px;">{t}</span>'
        for t, color in TYPE_COLORS.items()
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{title}</title>
{vis_script}
<style>
  body {{ margin: 0; font-family: sans-serif; background: #1e1e1e; color: #eee; }}
  #header {{ padding: 10px 16px; background: #2d2d2d; border-bottom: 1px solid #444; }}
  #header h2 {{ margin: 0 0 6px; font-size: 15px; }}
  #legend {{ margin-top: 4px; }}
  #stats {{ float: right; font-size: 13px; color: #aaa; margin-top: 2px; }}
  #network {{ width: 100%; height: calc(100vh - 90px); }}
</style>
</head>
<body>
<div id="header">
  <div id="stats">text {len(nodes)}text &nbsp;|&nbsp; text {len(edges)}text</div>
  <h2>{title}</h2>
  <div id="legend">{type_legend}</div>
</div>
<div id="network"></div>
<script>
const nodes = new vis.DataSet({nodes_json});
const edges = new vis.DataSet({edges_json});

const container = document.getElementById("network");
const network = new vis.Network(container, {{ nodes, edges }}, {{
  nodes: {{
    shape: "dot",
    size: 16,
    font: {{ color: "#eee", size: 13 }},
    borderWidth: 2,
    borderWidthSelected: 3,
  }},
  edges: {{
    font: {{ color: "#ccc", size: 11, align: "middle" }},
    color: {{ color: "#666", highlight: "#666" }},
    selectionWidth: 0,
    smooth: {{ type: "curvedCW", roundness: 0.1 }},
    width: 1.5,
  }},
  physics: {{
    stabilization: {{ iterations: 150 }},
    barnesHut: {{ gravitationalConstant: -20000, springLength: 200 }},
  }},
  interaction: {{ hover: true, tooltipDelay: 100 }},
}});
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="standardized.json → HTML text text")
    parser.add_argument("input", help="*_standardized.json text or output text")
    parser.add_argument("--output", "-o", default="", help="text HTML text (text: text text text)")
    args = parser.parse_args()

    json_path = _find_standardized(args.input)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # English comment.
    standardized = data if isinstance(data, list) else [data]
    nodes, edges = build_graph(standardized)
    stem = json_path.stem.replace("_standardized", "").replace("standardized", "").strip("_") or json_path.parent.name
    title = stem
    html = render_html(nodes, edges, title)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = json_path.parent / f"{stem}_graph.html"
    out_path.write_text(html, encoding="utf-8")

    print(f"completed: {out_path}")
    print(f"text: {len(nodes)}text, text: {len(edges)}text")


if __name__ == "__main__":
    main()
