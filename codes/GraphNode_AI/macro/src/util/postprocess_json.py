import json
import sys
import time
from collections import defaultdict
import os

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def post_process_clusters(input_file: str, output_file: str, verbose: bool = True):
    """English documentation."""
    # English comment.
    if not os.path.exists(input_file):
        print(f"text: text text '{input_file}'text text text text.")
        return

    if verbose:
        print(f"text text '{input_file}'text processing...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]

    # English comment.
    if "metadata" not in data:
        data["metadata"] = {}
    if "edge_statistics" not in data["metadata"]:
        data["metadata"]["edge_statistics"] = {}

    # English comment.
    cluster_id_to_name = {}
    if "clusters" in data["metadata"]:
        cluster_id_to_name.update(
            {
                cluster_id: cluster_info.get("name")
                for cluster_id, cluster_info in data["metadata"]["clusters"].items()
                if isinstance(cluster_info, dict)
            }
        )
    for node in nodes:
        if node.get("cluster_name") and node["cluster_id"] not in cluster_id_to_name:
            cluster_id_to_name[node["cluster_id"]] = node["cluster_name"]

    # English comment.
    # English comment.
    node_to_cluster = {node["id"]: node["cluster_id"] for node in nodes}
    initial_moves_made = 0
    total_moves = 0

    iteration_count = 0
    while True:
        iteration_count += 1
        moves_made = 0
        print(f"\n--- text {iteration_count} started ---")
        _iter_start = time.time()

        # English comment.
        node_iter = (
            _tqdm(nodes, total=len(nodes), desc=f"Merging iter {iteration_count}", file=sys.stdout, dynamic_ncols=True)
            if _tqdm else nodes
        )
        for node in node_iter:
            node_id = node["id"]
            current_cluster = node_to_cluster[node_id]

            # English comment.
            cluster_edge_counts = defaultdict(int)

            # English comment.
            for edge in edges:
                if edge["source"] == node_id:
                    neighbor_id = edge["target"]
                elif edge["target"] == node_id:
                    neighbor_id = edge["source"]
                else:
                    continue

                # English comment.
                neighbor_cluster = node_to_cluster[neighbor_id]
                cluster_edge_counts[neighbor_cluster] += 1

            # English comment.
            if not cluster_edge_counts:
                # English comment.
                continue

            # English comment.
            best_cluster = max(cluster_edge_counts, key=cluster_edge_counts.get)

            intra_cluster_edges = cluster_edge_counts.get(current_cluster, 0)
            max_inter_cluster_edges = cluster_edge_counts[best_cluster]

            # English comment.
            if (
                best_cluster != current_cluster
                and max_inter_cluster_edges > intra_cluster_edges
            ):
                if verbose:
                    print(
                        f"  - text '{node_id}'text text {current_cluster}text text {best_cluster}text text "
                        f"(text: {max_inter_cluster_edges} vs text text {intra_cluster_edges})"
                    )
                node_to_cluster[node_id] = best_cluster
                moves_made += 1
                if iteration_count == 1:
                    initial_moves_made += 1

        _iter_elapsed = time.time() - _iter_start
        pct = min(99, iteration_count * 20)
        _eta = int(_iter_elapsed)
        print(f"PROGRESS:merging:{pct}:{iteration_count}:0:{_eta}", flush=True)

        if moves_made == 0:
            if verbose:
                print("\n--- completed ---")
                print("text text text text text. text text text.")
            break
        else:
            total_moves += moves_made
            if verbose:
                print(
                    f"text {iteration_count} completed. text text text text text: {moves_made}"
                )

    # English comment.
    for node in data["nodes"]:
        new_cluster = node_to_cluster[node["id"]]
        node["cluster_id"] = new_cluster
        if cluster_id_to_name.get(new_cluster):
            node["cluster_name"] = cluster_id_to_name[new_cluster]

    # English comment.
    if "clusters" in data["metadata"]:
        cluster_sizes = defaultdict(int)
        for cluster in node_to_cluster.values():
            cluster_sizes[cluster] += 1
        for cluster_id, cluster_info in data["metadata"]["clusters"].items():
            if isinstance(cluster_info, dict):
                cluster_info["size"] = cluster_sizes.get(cluster_id, 0)

    # English comment.
    intra_cluster_edges = 0
    inter_cluster_edges = 0
    for edge in edges:
        source_cluster = node_to_cluster.get(edge["source"])
        target_cluster = node_to_cluster.get(edge["target"])
        if source_cluster is not None and target_cluster is not None:
            if source_cluster == target_cluster:
                intra_cluster_edges += 1
            else:
                inter_cluster_edges += 1

    # English comment.
    data["metadata"]["edge_statistics"]["intra_cluster_edges"] = intra_cluster_edges
    data["metadata"]["edge_statistics"]["inter_cluster_edges"] = inter_cluster_edges
    data["metadata"]["post_processing"] = {
        "iterations": iteration_count,
        "total_nodes_moved": total_moves,
    }

    # English comment.
    if verbose:
        print(f"\ntext text '{output_file}'text processing...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"text {total_moves}text text text.")
        print("completed.")


# English comment.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-process graph clusters for better cohesion."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input graph JSON file path."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output graph JSON file path."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output."
    )
    args = parser.parse_args()

    post_process_clusters(args.input, args.output, args.verbose)
