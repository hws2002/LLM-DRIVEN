"""Load notion.json into InputData + extract group_hints from page tree."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from util.io_schemas import InputData, Section, SourceNode


# English comment.
GroupHints = Dict[str, List[str]]


def load_notion(path: Path, min_children: int = 2) -> tuple[InputData, GroupHints, Set[str]]:
    """English documentation."""
    raw = json.loads(path.read_text(encoding="utf-8"))

    source_nodes, empty_node_ids = _parse_source_nodes_with_empty(raw.get("source_nodes", []))
    group_hints = _build_group_hints(raw.get("tree", {}).get("nodes", []), min_children)

    return InputData(source_nodes=source_nodes), group_hints, empty_node_ids


def _parse_source_nodes_with_empty(
    raw_nodes: List[Dict[str, Any]]
) -> tuple[List[SourceNode], Set[str]]:
    """English documentation."""
    nodes = []
    empty_node_ids: Set[str] = set()

    for node_dict in raw_nodes:
        real_sections = [
            Section(
                id=s.get("id", f"sec_{i}"),
                content=s.get("content", ""),
                section_title=s.get("section_title"),
            )
            for i, s in enumerate(node_dict.get("sections", []))
            if s.get("content", "").strip()
        ]

        nid = node_dict["id"]
        if not real_sections:
            # English comment.
            title = node_dict.get("title", "")
            if not title:
                continue
            sections = [Section(id=f"{nid}#title", content=title)]
            empty_node_ids.add(nid)
        else:
            sections = real_sections

        nodes.append(
            SourceNode(
                id=nid,
                title=node_dict.get("title"),
                sections=sections,
                source_type="notion",
                create_time=node_dict.get("create_time"),
                update_time=node_dict.get("update_time"),
            )
        )
    return nodes, empty_node_ids


def _build_group_hints(tree_nodes: List[Dict[str, Any]], min_children: int) -> GroupHints:
    """English documentation."""
    children_map: Dict[str, List[str]] = {}
    for node in tree_nodes:
        parent_id = node.get("parent_id")
        if not parent_id:
            continue
        children_map.setdefault(parent_id, []).append(node["id"])

    return {
        parent: children
        for parent, children in children_map.items()
        if len(children) >= min_children
    }


def save_group_hints(group_hints: GroupHints, output_path: Path) -> None:
    """English documentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(group_hints, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def label_subclusters_inline(graph: dict, language: str = "ko", verbose: bool = False) -> None:
    """English documentation."""
    import os
    try:
        from openai import OpenAI
    except ImportError:
        return

    api_key = (os.getenv("DEV_GROQ_API_KEY") or os.getenv("GROQ_API_KEY") or
               os.getenv("DEV_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", ""))
    base_url = "https://api.groq.com/openai/v1" if (
        os.getenv("DEV_GROQ_API_KEY") or os.getenv("GROQ_API_KEY")) else None
    model = "llama-3.1-8b-instant" if base_url else "gpt-4o-mini"

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception:
        return

    nodes = graph.get("nodes", [])
    node_map = {n["id"]: n for n in nodes}
    subclusters = graph.get("subclusters", [])
    lang_hint = {"ko": "Korean", "zh": "Chinese", "en": "English"}.get(language, "English")

    for sc in subclusters:
        if sc.get("name"):
            continue
        sc_nodes = [node_map[nid] for nid in sc.get("node_ids", []) if nid in node_map]
        cluster_name = (sc_nodes[0].get("cluster_name") or "") if sc_nodes else ""
        node_lines = "\n".join(
            f"- {n.get('title','?')}: {', '.join((n.get('top_keywords') or [])[:4])}"
            for n in sc_nodes[:8]
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content":
                     "You are a knowledge graph labeler. Generate a concise label (2-5 words) "
                     "for the subcluster. Respond with ONLY the label."},
                    {"role": "user", "content":
                     f"Cluster: {cluster_name}\nLanguage: {lang_hint}\nNodes:\n{node_lines}\nLabel:"},
                ],
                temperature=0.3, max_tokens=20,
            )
            name = resp.choices[0].message.content.strip().strip('"').strip("'")
            sc["name"] = name
            # English comment.
            keywords = sc.get("top_keywords", [])
            sc["top_keywords"] = [name] + [k for k in keywords if k != name]
            if verbose:
                print(f"   subcluster {sc['id']} → '{name}'")
        except Exception:
            pass  # English comment.


def inject_hierarchical_edges(graph: dict, notion_json_path: Path) -> None:
    """English documentation."""
    notion_data = json.loads(notion_json_path.read_text(encoding="utf-8"))
    tree_nodes = notion_data.get("tree", {}).get("nodes", [])
    nodes = graph.get("nodes", [])

    # uuid → node id
    uuid_to_node_id: Dict[str, int] = {}
    for n in nodes:
        orig = n.get("orig_id", "")
        parts = orig.split("_", 1)
        uuid = parts[1] if len(parts) == 2 else orig
        uuid_to_node_id[uuid] = n["id"]

    added = 0
    existing_edges = graph.get("edges", [])
    existing_pairs = {(e["source"], e["target"]) for e in existing_edges}

    for tn in tree_nodes:
        child_uuid = tn["id"]
        parent_uuid = tn.get("parent_id")
        if not parent_uuid:
            continue
        child_nid = uuid_to_node_id.get(child_uuid)
        parent_nid = uuid_to_node_id.get(parent_uuid)
        if child_nid is None or parent_nid is None:
            continue
        if (parent_nid, child_nid) in existing_pairs:
            continue
        existing_edges.append({
            "source": parent_nid,
            "target": child_nid,
            "weight": 1.0,
            "type": "hierarchical",
            "intraCluster": False,
        })
        existing_pairs.add((parent_nid, child_nid))
        added += 1

    graph["edges"] = existing_edges
    if added:
        print(f"   [notion] {added} hierarchical edges added to graph_final.json")


def apply_bottomup_cluster_from_embeddings(
    cluster_path: Path,
    features_path: Path,
    notion_json_path: Path,
    output_path: Path,
    empty_node_ids: Optional[Set[str]] = None,
    verbose: bool = False,
) -> None:
    """English documentation."""
    clusters_data = json.loads(cluster_path.read_text(encoding="utf-8"))
    features_data = json.loads(features_path.read_text(encoding="utf-8"))
    notion_data = json.loads(notion_json_path.read_text(encoding="utf-8"))

    assignments = clusters_data.get("assignments", [])
    conversations = features_data.get("conversations", [])

    # English comment.
    embeddings = features_data.get("embeddings", [])
    orig_id_to_emb: Dict[str, np.ndarray] = {}
    for i, conv in enumerate(conversations):
        oid = conv.get("orig_id", "")
        if oid and i < len(embeddings) and embeddings[i]:
            orig_id_to_emb[oid] = np.array(embeddings[i], dtype=np.float32)

    # English comment.
    orig_id_to_idx: Dict[str, int] = {a["orig_id"]: i for i, a in enumerate(assignments)}
    # English comment.
    uuid_to_orig_id: Dict[str, str] = {}
    for oid in orig_id_to_idx:
        # "src0_3f67c3e5-..." → "3f67c3e5-..."
        parts = oid.split("_", 1)
        uuid = parts[1] if len(parts) == 2 else oid
        uuid_to_orig_id[uuid] = oid

    # English comment.
    _empty_uuids = empty_node_ids or set()

    def has_content(orig_id: str) -> bool:
        # English comment.
        parts = orig_id.split("_", 1)
        uuid = parts[1] if len(parts) == 2 else orig_id
        if uuid in _empty_uuids:
            return False
        emb = orig_id_to_emb.get(orig_id)
        if emb is None:
            return False
        return float(np.linalg.norm(emb)) > 1e-6

    # English comment.
    cluster_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
    for a in assignments:
        oid = a["orig_id"]
        if has_content(oid) and oid in orig_id_to_emb:
            cluster_vecs[a["cluster_id"]].append(orig_id_to_emb[oid])

    centroids: Dict[str, np.ndarray] = {}
    for cid, vecs in cluster_vecs.items():
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        centroids[cid] = centroid / norm if norm > 1e-6 else centroid

    if not centroids:
        if verbose:
            print("   [notion bottomup] centroids text — text")
        return

    # English comment.
    tree_nodes = notion_data.get("tree", {}).get("nodes", [])
    children_map: Dict[str, List[str]] = defaultdict(list)
    all_ids: Set[str] = set()
    for tn in tree_nodes:
        nid = tn["id"]
        all_ids.add(nid)
        pid = tn.get("parent_id")
        if pid:
            children_map[pid].append(nid)

    # English comment.
    roots = [tn["id"] for tn in tree_nodes if not tn.get("parent_id")]
    order: List[str] = []
    queue = deque(roots)
    visited: Set[str] = set()
    while queue:
        nid = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        order.append(nid)
        for child in children_map.get(nid, []):
            queue.append(child)

    # English comment.
    reassigned = 0
    node_emb_cache: Dict[str, np.ndarray] = {}  # English comment.

    for nid in reversed(order):
        # English comment.
        orig_id = uuid_to_orig_id.get(nid)

        # English comment.
        if orig_id and has_content(orig_id):
            if orig_id in orig_id_to_emb:
                node_emb_cache[nid] = orig_id_to_emb[orig_id]
            continue

        # English comment.
        child_embs = [
            node_emb_cache[child]
            for child in children_map.get(nid, [])
            if child in node_emb_cache
        ]
        if not child_embs:
            continue

        avg_emb = np.mean(child_embs, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm < 1e-6:
            continue
        avg_emb = avg_emb / norm
        node_emb_cache[nid] = avg_emb

        # English comment.
        best_cluster = max(centroids, key=lambda cid: float(np.dot(avg_emb, centroids[cid])))

        # English comment.
        full_orig_id = uuid_to_orig_id.get(nid)
        if full_orig_id and full_orig_id in orig_id_to_idx:
            idx = orig_id_to_idx[full_orig_id]
            old = assignments[idx]["cluster_id"]
            if old != best_cluster:
                assignments[idx]["cluster_id"] = best_cluster
                assignments[idx]["_notion_bottomup_reassigned"] = True
                reassigned += 1

    if verbose:
        print(f"   [notion bottomup] {reassigned}text text text → embedding text cluster text")

    clusters_data["assignments"] = assignments
    output_path.write_text(json.dumps(clusters_data, ensure_ascii=False, indent=2), encoding="utf-8")


def apply_group_hints_to_clusters(
    cluster_path: Path,
    group_hints_path: Path,
    output_path: Path,
    verbose: bool = False,
) -> None:
    """English documentation."""
    from collections import Counter

    clusters_data = json.loads(cluster_path.read_text(encoding="utf-8"))
    group_hints: GroupHints = json.loads(group_hints_path.read_text(encoding="utf-8"))

    assignments = clusters_data.get("assignments", [])
    # English comment.
    orig_id_to_idx = {a["orig_id"]: i for i, a in enumerate(assignments)}
    # English comment.
    uuid_to_orig_id = {}
    for orig_id in orig_id_to_idx:
        parts = orig_id.split("_", 1)
        uuid = parts[1] if len(parts) == 2 else orig_id
        uuid_to_orig_id[uuid] = orig_id

    reassigned = 0
    for parent_id, child_ids in group_hints.items():
        # English comment.
        group_assignments = [
            (uuid_to_orig_id[child_id], assignments[orig_id_to_idx[uuid_to_orig_id[child_id]]]["cluster_id"])
            for child_id in child_ids
            if child_id in uuid_to_orig_id
        ]
        if len(group_assignments) < 2:
            continue

        # English comment.
        counter = Counter(cluster_id for _, cluster_id in group_assignments)
        majority_cluster = counter.most_common(1)[0][0]

        for orig_id, cluster_id in group_assignments:
            if cluster_id != majority_cluster:
                idx = orig_id_to_idx[orig_id]
                assignments[idx]["cluster_id"] = majority_cluster
                assignments[idx]["_notion_group_reassigned"] = True
                reassigned += 1

    if verbose and reassigned:
        print(f"   [notion] group_hints: {reassigned} node(s) reassigned to majority cluster")

    clusters_data["assignments"] = assignments
    output_path.write_text(
        json.dumps(clusters_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
