"""End-to-end pipeline orchestration for conversation graph building."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from merge_graph import merge_graph_data
from util.graph_utils import convert_to_frontend_format
from util.postprocess_json import post_process_clusters
import extract_features as _extract_features_mod
import cluster_with_llm as _cluster_mod
import build_edges as _build_edges_mod
import build_subclusters as _build_subclusters_mod

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent  # macro/ directory
REPO_ROOT = PROJECT_ROOT.parent  # GraphNode_AI/ directory



def _script_path(filename: str) -> str:
    """Return an absolute path to a sibling step script."""
    return str(SRC_DIR / filename)


def run_step(cmd: List[str], step_name: str, verbose: bool = True, env: Optional[Dict[str, str]] = None) -> bool:
    """
    Execute a subprocess command for a pipeline step.

    Args:
        cmd: Command to execute as list of strings
        step_name: Descriptive name for the step
        verbose: Whether to print progress messages
        env: Optional environment variables to set

    Returns:
        True if command succeeded (returncode == 0), False otherwise
    """
    if verbose:
        print(f"\n Running {step_name}...")
        print(f"   Command: {' '.join(cmd)}")

    # Set up environment with PYTHONPATH for src modules
    if env is None:
        env = os.environ.copy()
        pythonpath = os.pathsep.join([str(PROJECT_ROOT), str(REPO_ROOT)])
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = pythonpath

    try:
        # Run command and capture output in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            env=env,
        )

        # Print output in real-time
        if process.stdout:
            for line in process.stdout:
                print(line, end="")

        # Wait for process to complete
        return_code = process.wait()

        if return_code == 0:
            if verbose:
                print(f" {step_name} completed successfully")
            return True
        else:
            print(f" {step_name} failed with return code {return_code}")
            sys.exit(1)

    except Exception as exc:
        print(f" {step_name} failed with exception: {exc}")
        sys.exit(1)


def validate_file_exists(path: Path, description: str) -> None:
    """
    Validate that a file exists.

    Args:
        path: Path to check
        description: Human-readable description of the file

    Raises:
        SystemExit if file does not exist
    """
    if not path.exists():
        print(f" {description} not found: {path}")
        sys.exit(1)


def main() -> None:
    """Main pipeline orchestration."""
    parser = argparse.ArgumentParser(
        description="End-to-end conversation graph building pipeline"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input chat history JSON",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        help="Fixed number of clusters (optional)",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=3,
        help="Min clusters (if num-clusters not set, default: 3)",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Max clusters (if num-clusters not set, default: 8)",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.8,
        help="High confidence edge threshold (default: 0.8)",
    )
    parser.add_argument(
        "--medium-threshold",
        type=float,
        default=0.6,
        help="Medium confidence edge threshold (default: 0.6)",
    )
    parser.add_argument(
        "--no-llm-edges",
        action="store_true",
        help="Skip LLM verification for edges",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="LLM provider for clustering (default: openai)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="LLM model for clustering (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--skip-indexing",
        action="store_true",
        help="Skip embedding indexing step (Step 7)",
    )
    parser.add_argument(
        "--skip-subclustering",
        action="store_true",
        help="Skip sub-clustering step (Step 6)",
    )
    parser.add_argument(
        "--subcluster-method",
        type=str,
        choices=["louvain", "components", "cliques"],
        default="louvain",
        help="Sub-clustering method (default: louvain)",
    )
    parser.add_argument(
        "--subcluster-resolution",
        type=float,
        default=1.0,
        help="Resolution for Louvain sub-clustering (default: 1.0)",
    )
    parser.add_argument(
        "--subcluster-min-size",
        type=int,
        default=2,
        help="Minimum nodes per sub-cluster (default: 2)",
    )
    parser.add_argument(
        "--vector-db-dir",
        type=Path,
        default=None,
        help="Custom vector DB directory (default: output_dir/vector_db)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="",
        help="User ID for ChromaDB record IDs (e.g. 'user123' → 'user123_conv_0')",
    )
    parser.add_argument(
        "--input-type",
        choices=["auto", "chatgpt", "markdown"],
        default="auto",
        help="Input source type. 'auto' detects from file extension/type (default: auto)",
    )
    parser.add_argument(
        "--notion-token",
        type=str,
        default=None,
        help="Notion API integration token (for --input-type notion, Phase 5)",
    )
    parser.add_argument(
        "--extra-input",
        type=Path,
        action="append",
        default=[],
        dest="extra_inputs",
        help="Additional input source to merge (can be used multiple times). "
             "Supports .json (ChatGPT) or directory/.md (markdown).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "ko", "zh"],
        help="Output language for cluster names (default: en)",
    )

    args = parser.parse_args()

    # Validate input files
    input_path = args.input.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()

    # Validate input (can be file or directory for markdown)
    if not input_path.exists():
        print(f" Input not found: {input_path}")
        sys.exit(1)

    validate_file_exists(config_path, "Config YAML file")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # English comment.
    group_hints_path: Optional[Path] = None
    if input_path.is_dir():
        notion_json = input_path / "notion.json"
        if notion_json.exists():
            from util.notion_loader import load_notion, save_group_hints
            if args.verbose:
                print(f" Notion bundle detected: {notion_json}")
            notion_data, group_hints, empty_node_ids = load_notion(notion_json)
            # English comment.
            _empty_nodes_path = output_dir / "_notion_empty_nodes.json"
            _empty_nodes_path.write_text(
                json.dumps(list(empty_node_ids), ensure_ascii=False), encoding="utf-8"
            )
            if group_hints:
                group_hints_path = output_dir / "_notion_group_hints.json"
                save_group_hints(group_hints, group_hints_path)
                if args.verbose:
                    print(f"   └─ {len(group_hints)} group hint(s) saved → {group_hints_path}")

    # Handle multi-source input merging if --extra-input is specified
    original_input_path = input_path
    if args.extra_inputs:
        from extract_features import load_messages, merge_inputs

        if args.verbose:
            print("=" * 60)
            print(" Merging multiple input sources...")
            print("=" * 60)

        # Load primary input
        primary_data = load_messages(input_path)
        primary_nodes = getattr(primary_data, 'source_nodes', None) or getattr(primary_data, 'conversations', [])

        if args.verbose:
            print(f"   Primary input: {input_path}")
            print(f"     └─ Loaded {len(primary_nodes)} nodes")

        # Load and merge extra inputs
        all_data = [primary_data]
        for extra_path in args.extra_inputs:
            extra_path_resolved = extra_path.resolve()
            if not extra_path_resolved.exists():
                print(f" Extra input not found: {extra_path_resolved}")
                sys.exit(1)

            extra_data = load_messages(extra_path_resolved)
            extra_nodes = getattr(extra_data, 'source_nodes', None) or getattr(extra_data, 'conversations', [])

            if args.verbose:
                print(f"  + Extra input: {extra_path_resolved}")
                print(f"     └─ Loaded {len(extra_nodes)} nodes")

            all_data.append(extra_data)

        # Merge all data sources
        merged_data = merge_inputs(*all_data)
        merged_nodes = getattr(merged_data, 'source_nodes', None) or getattr(merged_data, 'conversations', [])

        if args.verbose:
            print(f"  → Merged total: {len(merged_nodes)} nodes")

            # Show source type distribution
            from collections import Counter
            source_types = Counter(getattr(node, 'source_type', 'unknown') for node in merged_nodes)
            print(f"     Source distribution: {dict(source_types)}")
            print("=" * 60)

        # Save merged data as temporary JSON for subprocess pipeline
        merged_tmp_path = output_dir / "_merged_input.json"

        # Convert merged data to JSON format that extract_features.py expects
        merged_json = {
            "source_nodes": [
                {
                    "id": node.id,
                    "title": node.title,
                    "sections": [{"content": s.content, "role": s.role, "section_title": s.section_title} for s in node.sections],
                    "source_type": node.source_type,
                    "create_time": node.create_time,
                    "update_time": node.update_time,
                }
                for node in merged_nodes
            ]
        }

        with open(merged_tmp_path, 'w', encoding='utf-8') as f:
            json.dump(merged_json, f, ensure_ascii=False, indent=2)

        # Update input_path to point to merged file
        input_path = merged_tmp_path

    if args.verbose:
        print("=" * 60)
        print(" Starting Conversation Graph Building Pipeline")
        print("=" * 60)
        print(f"Input: {original_input_path}")
        if args.extra_inputs:
            print(f"       (merged with {len(args.extra_inputs)} extra source(s))")
        print(f"Config: {config_path}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

    pipeline_start = time.perf_counter()

    # Define output paths
    features_path = output_dir / "features.json"
    cluster_path = output_dir / "clusters.json"
    edges_path = output_dir / "edges.json"
    final_graph_path = output_dir / "graph.json"
    postprocessed_graph_path = output_dir / "graph_postprocessed.json"
    subclusters_path = output_dir / "subclusters.json"
    final_output_path = output_dir / "graph_final.json"  # With subclusters integrated

    # Step 1: Extract keywords and embeddings
    print(f"\n{'='*60}\n Running Step 1: Keyword & Embedding Extraction\n{'='*60}\n")
    _extract_features_mod.main([
        "--in", str(input_path),
        "--out", str(features_path),
        "--cfg", str(config_path),
    ])

    # Validate feature extraction results
    validate_file_exists(features_path, "Feature data JSON")

    with open(features_path, "r", encoding="utf-8") as f:
        features_data = json.load(f)
    timing = features_data.get("metadata", {}).get("timing", {})
    step1_total = float(timing.get("total_seconds", 0.0) or 0.0)
    step1_embedding = float(timing.get("embedding_seconds", 0.0) or 0.0)
    step1_keyword = float(timing.get("keyword_seconds", 0.0) or 0.0)

    print(f" Step 1 completed in {step1_total:.1f}s")
    print(f"  └─ Embedding: {step1_embedding:.1f}s, Keyword: {step1_keyword:.1f}s\n")
    # Step 2: LLM-based clustering
    print(f"\n{'='*60}\n Running Step 2: LLM-based Clustering\n{'='*60}\n")
    print("STEP_START:clustering", flush=True)
    cluster_argv = [
        "--input", str(features_path),
        "--output", str(cluster_path),
        "--provider", args.provider,
        "--model", args.model,
        "--language", args.language,
    ]
    if args.num_clusters:
        cluster_argv.extend(["--num-clusters", str(args.num_clusters)])
    else:
        cluster_argv.extend(["--min-clusters", str(args.min_clusters), "--max-clusters", str(args.max_clusters)])
    if args.verbose:
        cluster_argv.append("--verbose")

    step2_start = time.perf_counter()
    _cluster_mod.main(cluster_argv)
    step2_time = time.perf_counter() - step2_start
    print("STEP_DONE:clustering", flush=True)

    # Validate cluster results
    validate_file_exists(cluster_path, "Cluster assignments JSON")

    # English comment.
    _notion_json = (original_input_path / "notion.json") if original_input_path.is_dir() else None
    _empty_nodes_path = output_dir / "_notion_empty_nodes.json"
    if _notion_json and _notion_json.exists() and features_path.exists():
        empty_node_ids = set(json.loads(_empty_nodes_path.read_text(encoding="utf-8"))) if _empty_nodes_path.exists() else set()
        from util.notion_loader import apply_bottomup_cluster_from_embeddings
        apply_bottomup_cluster_from_embeddings(
            cluster_path, features_path, _notion_json, cluster_path,
            empty_node_ids=empty_node_ids, verbose=args.verbose
        )

    # English comment.
    if group_hints_path and group_hints_path.exists():
        from util.notion_loader import apply_group_hints_to_clusters
        apply_group_hints_to_clusters(cluster_path, group_hints_path, cluster_path, verbose=args.verbose)

    # Step 3: Build edges
    print(f"\n{'='*60}\n Running Step 3: Edge Generation\n{'='*60}\n")
    print("STEP_START:edges", flush=True)
    edge_argv = [
        "--intermediate", str(features_path),
        "--clusters", str(cluster_path),
        "--output", str(edges_path),
        "--high-threshold", str(args.high_threshold),
        "--medium-threshold", str(args.medium_threshold),
    ]
    if args.no_llm_edges:
        edge_argv.append("--no-llm")
    if args.verbose:
        edge_argv.append("--verbose")

    step3_start = time.perf_counter()
    _build_edges_mod.main(edge_argv)
    step3_time = time.perf_counter() - step3_start
    print("PROGRESS:edges:100:1:1:0", flush=True)
    print("STEP_DONE:edges", flush=True)

    # Validate edge results
    validate_file_exists(edges_path, "Edges JSON")

    # Step 4: Merge results into final graph
    print(f"\n{'='*60}")
    print(f" Step 4: Merging final graph...")
    print("STEP_START:merging", flush=True)
    print(f"{'='*60}\n")

    step4_start = time.perf_counter()
    merge_graph_data(
        features_path=features_path,
        cluster_path=cluster_path,
        edges_path=edges_path,
        output_path=final_graph_path,
        verbose=args.verbose,
    )
    step4_time = time.perf_counter() - step4_start

    print(f"\n Step 4 completed in {step4_time:.1f}s\n")
    validate_file_exists(final_graph_path, "Merged graph JSON")

    # Step 5: Post-process clusters
    print(f"\n{'='*60}")
    print(f" Step 5: Post-processing and optimizing clusters...")
    print(f"{'='*60}\n")

    step5_start = time.perf_counter()
    post_process_clusters(
        input_file=str(final_graph_path),
        output_file=str(postprocessed_graph_path),
        verbose=args.verbose,
    )
    step5_time = time.perf_counter() - step5_start

    print(f"\n Step 5 completed in {step5_time:.1f}s\n")
    validate_file_exists(postprocessed_graph_path, "Post-processed graph JSON")

    # Step 6: Sub-clustering (optional)
    step6_time = 0.0
    if not args.skip_subclustering:
        print(f"\n{'='*60}")
        print(f" Step 6: Building sub-clusters...")
        print(f"{'='*60}\n")

        print(f"\n{'='*60}\n Running Step 6: Sub-clustering\n{'='*60}\n")
        subcluster_argv = [
            "--graph", str(postprocessed_graph_path),
            "--output", str(subclusters_path),
            "--method", args.subcluster_method,
            "--min-size", str(args.subcluster_min_size),
            "--resolution", str(args.subcluster_resolution),
        ]
        if not args.verbose:
            subcluster_argv.append("--quiet")

        step6_start = time.perf_counter()
        _build_subclusters_mod.main(subcluster_argv)
        step6_time = time.perf_counter() - step6_start

        print(f"\n Step 6 completed in {step6_time:.1f}s\n")

    # Step 6.5: Integrate subclusters into final graph
    print(f"\n{'='*60}")
    print(f" Step 6.5: Integrating subclusters into final graph...")
    print(f"{'='*60}\n")

    # Load postprocessed graph (which has the cluster reassignments from Step 5)
    with open(postprocessed_graph_path, "r", encoding="utf-8") as f:
        final_graph = json.load(f)

    # Load and integrate subclusters if available
    subcluster_data = {}
    if not args.skip_subclustering and subclusters_path.exists():
        with open(subclusters_path, "r", encoding="utf-8") as f:
            subcluster_data = json.load(f)

        # Build node -> subcluster mapping
        node_to_subcluster = {
            int(k): v for k, v in subcluster_data.get("node_to_subcluster", {}).items()
        }

        # Add subcluster_id to each node
        for node in final_graph.get("nodes", []):
            node_id = node.get("id")
            node["subcluster_id"] = node_to_subcluster.get(node_id)

        # Add subclusters array to graph
        final_graph["subclusters"] = subcluster_data.get("subclusters", [])

        # Add subcluster statistics to metadata
        sc_metadata = subcluster_data.get("metadata", {})
        final_graph["metadata"]["subcluster_statistics"] = {
            "total_subclusters": sc_metadata.get("total_subclusters", 0),
            "nodes_in_subclusters": sc_metadata.get("total_nodes_in_subclusters", 0),
            "coverage": sc_metadata.get("coverage", 0),
            "method": sc_metadata.get("method", "unknown"),
            "parameters": sc_metadata.get("parameters", {}),
            "cluster_breakdown": sc_metadata.get("cluster_stats", {}),
        }
        final_graph["metadata"]["total_subclusters"] = sc_metadata.get(
            "total_subclusters", 0
        )

        if args.verbose:
            print(
                f"    Integrated {len(subcluster_data.get('subclusters', []))} subclusters"
            )
    else:
        # No subclusters - add empty subcluster_id to nodes
        for node in final_graph.get("nodes", []):
            node["subcluster_id"] = None
        final_graph["subclusters"] = []
        final_graph["metadata"]["subcluster_statistics"] = {}
        final_graph["metadata"]["total_subclusters"] = 0

    # Save final graph
    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(final_graph, f, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"    Saved final graph to {final_output_path}")

    # English comment.
    from util.notion_loader import label_subclusters_inline
    label_subclusters_inline(final_graph, language=args.language, verbose=args.verbose)

    # English comment.
    if _notion_json and _notion_json.exists():
        from util.notion_loader import inject_hierarchical_edges
        inject_hierarchical_edges(final_graph, _notion_json)

    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(final_graph, f, ensure_ascii=False, indent=2)

    # Generate frontend graph
    frontend_output_path = output_dir / "frontend_graph.json"
    frontend_graph = convert_to_frontend_format(final_graph, subcluster_data)
    with open(frontend_output_path, "w", encoding="utf-8") as f:
        json.dump(frontend_graph, f, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"    Saved frontend graph to {frontend_output_path}")

    # Step 7: Index embeddings into vector database (optional)
    step7_time = 0.0
    if not args.skip_indexing:
        print(f"\n{'='*60}")
        print(f" Step 7: Indexing embeddings into vector database...")
        print(f"{'='*60}\n")

        # Determine vector DB directory
        vector_db_dir = (
            args.vector_db_dir if args.vector_db_dir else output_dir / "vector_db"
        )
        vector_db_dir.mkdir(parents=True, exist_ok=True)

        index_cmd = [
            sys.executable,
            "-m",
            "src.insights.index_embeddings",
            "--features",
            str(features_path),
            "--graph",
            str(final_output_path),
            "--output-dir",
            str(vector_db_dir),
        ]

        if args.user_id:
            index_cmd.extend(["--user-id", args.user_id])

        if args.verbose:
            index_cmd.append("--verbose")

        step7_start = time.perf_counter()
        run_step(
            index_cmd,
            "Step 7: Embedding Indexing",
            verbose=args.verbose,
        )
        step7_time = time.perf_counter() - step7_start

        print(f"\n Step 7 completed in {step7_time:.1f}s")
        print(f"  Vector DB location: {vector_db_dir.resolve()}\n")

    total_pipeline_time = time.perf_counter() - pipeline_start
    print("PROGRESS:merging:100:1:1:0", flush=True)
    print("STEP_DONE:merging", flush=True)
    print("STEP_START:done", flush=True)

    # Save timing.json
    timing_data = {
        "step1_total_seconds": round(step1_total, 2),
        "step1_embedding_seconds": round(step1_embedding, 2),
        "step1_keyword_seconds": round(step1_keyword, 2),
        "step2_clustering_seconds": round(step2_time, 2),
        "step3_edge_seconds": round(step3_time, 2),
        "step4_merge_seconds": round(step4_time, 2),
        "step5_postprocess_seconds": round(step5_time, 2),
        "step6_subcluster_seconds": round(step6_time, 2),
        "step7_indexing_seconds": round(step7_time, 2),
        "total_seconds": round(total_pipeline_time, 2),
    }
    timing_path = output_dir / "timing.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(" Pipeline Complete!")
    print(f"{'='*60}")
    print("\n Timing Summary:")
    print(f"  Step 1 (Feature Extraction):  {step1_total:.1f}s")
    print(f"    ├─ Embedding generation:    {step1_embedding:.1f}s")
    print(f"    └─ Keyword extraction:      {step1_keyword:.1f}s")
    print(f"  Step 2 (LLM Clustering):      {step2_time:.1f}s")
    print(f"  Step 3 (Edge Generation):     {step3_time:.1f}s")
    print(f"  Step 4 (Graph Merging):       {step4_time:.1f}s")
    print(f"  Step 5 (Cluster Post-process):{step5_time:.1f}s")
    if not args.skip_subclustering:
        print(f"  Step 6 (Sub-clustering):      {step6_time:.1f}s")
    if not args.skip_indexing:
        print(f"  Step 7 (Embedding Indexing):  {step7_time:.1f}s")
    print(f"  {'─'*40}")
    print(f"  Total Pipeline Time:          {total_pipeline_time:.1f}s")

    print("\n Final Graph Statistics:")

    # Load and display graph statistics
    with open(final_output_path, encoding="utf-8") as f:
        graph_data = json.load(f)
        metadata = graph_data.get("metadata", {})
        edge_stats = metadata.get("edge_statistics", {})
        subcluster_stats = metadata.get("subcluster_statistics", {})

        print(f"  Nodes:                  {metadata.get('total_nodes', 0)}")
        print(f"  Edges:                  {metadata.get('total_edges', 0)}")
        print(f"  Clusters:               {metadata.get('total_clusters', 0)}")

        if subcluster_stats:
            print(
                f"  Sub-clusters:           {subcluster_stats.get('total_subclusters', 0)}"
            )
            print(
                f"  Nodes in sub-clusters:  {subcluster_stats.get('nodes_in_subclusters', 0)} ({subcluster_stats.get('coverage', 0)*100:.1f}%)"
            )

        print(f"  Intra-cluster edges:    {edge_stats.get('intra_cluster_edges', 0)}")
        print(f"  Inter-cluster edges:    {edge_stats.get('inter_cluster_edges', 0)}")
        print(f"  Edge density:           {edge_stats.get('edge_density', 0):.4f}")

    print(f"\n Final graph saved to: {final_output_path.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
