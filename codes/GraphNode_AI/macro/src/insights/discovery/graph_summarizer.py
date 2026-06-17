"""Graph summarizer for generating insights from conversation knowledge graphs."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import List, Optional

from util.llm_clients import BaseLLMClient

from ..core import GraphLoader
from .schema import (
    ClusterAnalysis,
    ClusterConnection,
    GraphSummary,
    OverviewSection,
    Pattern,
    Recommendation,
)
from .prompts import (
    CLUSTER_ANALYSIS_PROMPT,
    ENHANCED_PATTERN_DETECTION_PROMPT,
    OVERVIEW_PROMPT,
    PATTERN_DETECTION_PROMPT,
    RECOMMENDATION_PROMPT,
)
from .prompts.summarizer_prompts import get_language_instruction
from .metrics import RepetitionMetricsCollector

logger = logging.getLogger(__name__)


class GraphSummarizer:
    """
    Analyzes knowledge graph structure and generates insights.
    Uses LLM to synthesize findings into natural language.
    """

    def __init__(
        self,
        graph_loader: GraphLoader,
        llm_client: BaseLLMClient,
        language: str = "en"
    ):
        """Initialize graph summarizer.

        Args:
            graph_loader: GraphLoader instance
            llm_client: LLM client for generating insights
            language: ISO 639-1 language code for LLM output
                (e.g. ``"ko"``, ``"en"``, ``"zh"``). Defaults to ``"en"``.
        """
        self.graph_loader = graph_loader
        self.llm_client = llm_client
        self.language = language
        self._lang_instruction = get_language_instruction(language)

    def generate_summary(
        self,
        detail_level: str = "standard",
        focus_areas: Optional[List[str]] = None,
        include_recommendations: bool = True
    ) -> GraphSummary:
        """Generate comprehensive graph summary.

        Args:
            detail_level: Level of detail ("brief", "standard", "detailed")
            focus_areas: Optional list of cluster IDs to focus on
            include_recommendations: Whether to include recommendations

        Returns:
            GraphSummary instance
        """
        logger.info(f"Generating graph summary (detail_level={detail_level})")

        # Get graph statistics
        stats = self.graph_loader.get_graph_stats()
        all_clusters = self.graph_loader.get_all_clusters()

        # Filter to focus areas if specified
        if focus_areas:
            clusters_to_analyze = [c for c in all_clusters if c.id in focus_areas]
        else:
            clusters_to_analyze = all_clusters

        # Generate overview
        logger.info("Generating overview...")
        overview = self._generate_overview(stats, all_clusters)

        # Analyze clusters
        logger.info(f"Analyzing {len(clusters_to_analyze)} clusters...")
        cluster_analyses = []
        total_clusters = len(clusters_to_analyze)
        _summary_start = time.time()
        for idx, cluster in enumerate(clusters_to_analyze):
            try:
                analysis = self.analyze_cluster(cluster.id)
                cluster_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Failed to analyze cluster {cluster.id}: {e}")
            pct = int((idx + 1) / total_clusters * 100) if total_clusters > 0 else 100
            _elapsed = time.time() - _summary_start
            _rate = (idx + 1) / _elapsed if _elapsed > 0 else 0
            _eta = int((total_clusters - idx - 1) / _rate) if _rate > 0 else 0
            print(f"PROGRESS:summary:{pct}:{idx + 1}:{total_clusters}:{_eta}", flush=True)

        # Identify patterns
        logger.info("Identifying patterns...")
        patterns = self.identify_patterns(detail_level)

        # Find connections
        logger.info("Finding cluster connections...")
        connections = self._find_cluster_connections(all_clusters)

        # Generate recommendations
        recommendations = []
        if include_recommendations:
            logger.info("Generating recommendations...")
            recommendations = self._generate_recommendations(
                overview,
                cluster_analyses,
                patterns
            )

        summary = GraphSummary(
            overview=overview,
            clusters=cluster_analyses,
            patterns=patterns,
            connections=connections,
            recommendations=recommendations,
            generated_at=datetime.now().isoformat(),
            detail_level=detail_level
        )

        logger.info("Graph summary generation complete")
        return summary

    def analyze_cluster(self, cluster_id: str) -> ClusterAnalysis:
        """Deep dive analysis of a specific cluster.

        Args:
            cluster_id: Cluster ID to analyze

        Returns:
            ClusterAnalysis instance
        """
        cluster_with_nodes = self.graph_loader.get_cluster_with_nodes(cluster_id)
        if not cluster_with_nodes:
            raise ValueError(f"Cluster {cluster_id} not found")

        cluster = cluster_with_nodes.cluster
        nodes = cluster_with_nodes.nodes

        # Compute metrics
        density = self._compute_cluster_density(cluster_id)
        centrality = self._compute_cluster_centrality(cluster_id)
        recency = self._compute_recency(cluster_id)

        # Get top keywords
        top_keywords = cluster_with_nodes.get_all_keywords(top_n=10)

        # Get sample conversation topics (first few nodes)
        sample_topics = []
        for node in nodes[:5]:
            keywords = node.get_keyword_terms(top_n=3)
            if keywords:
                sample_topics.append(", ".join(keywords))

        # Call LLM for analysis
        prompt = CLUSTER_ANALYSIS_PROMPT.format(
            cluster_name=cluster.name,
            cluster_description=cluster.description,
            cluster_size=cluster.size,
            density=density,
            centrality=centrality,
            recency=recency,
            top_keywords="\n".join(f"- {kw}" for kw in top_keywords[:10]),
            sample_topics="\n".join(f"- {topic}" for topic in sample_topics)
        )

        try:
            response = self.llm_client.call_llm(
                system_prompt=f"You are an expert at analyzing conversation patterns and knowledge graphs. {self._lang_instruction}",
                user_prompt=prompt,
                temperature=0.7
            )

            # Parse JSON response
            analysis_data = self._parse_json_response(response)

            return ClusterAnalysis(
                cluster_id=cluster_id,
                name=cluster.name,
                size=cluster.size,
                density=density,
                centrality=centrality,
                recency=recency,
                top_keywords=top_keywords[:5],
                key_themes=analysis_data.get("key_themes", []),
                common_question_types=analysis_data.get("common_question_types", []),
                insight_text=analysis_data.get("insight_text", ""),
                notable_conversations=[]  # Could be populated with evidence
            )

        except Exception as e:
            logger.error(f"LLM analysis failed for cluster {cluster_id}: {e}")
            # Return basic analysis without LLM insights
            return ClusterAnalysis(
                cluster_id=cluster_id,
                name=cluster.name,
                size=cluster.size,
                density=density,
                centrality=centrality,
                recency=recency,
                top_keywords=top_keywords[:5],
                key_themes=cluster.key_themes[:3] if cluster.key_themes else [],
                common_question_types=[],
                insight_text=f"A cluster of {cluster.size} conversations about {cluster.name}.",
                notable_conversations=[]
            )

    def identify_patterns(self, detail_level: str = "standard") -> List[Pattern]:
        """Identify recurring patterns across the graph.

        Args:
            detail_level: Level of detail for pattern detection

        Returns:
            List of identified patterns
        """
        stats = self.graph_loader.get_graph_stats()
        clusters = self.graph_loader.get_all_clusters()

        # Collect measured metrics
        repetition_collector = RepetitionMetricsCollector(self.graph_loader.graph_store)
        repetition_summary = repetition_collector.get_summary_for_llm()
        repetition_evidence = repetition_collector.find_evidence()

        logger.info(f"Collected {len(repetition_evidence)} pieces of repetition evidence")

        # Prepare cluster summaries
        cluster_summaries = []
        for cluster in clusters:
            cluster_summaries.append(
                f"- {cluster.name}: {cluster.size} conversations, themes: {', '.join(cluster.key_themes[:3])}"
            )

        # Get connections
        connections = self._find_cluster_connections(clusters)
        connection_desc = []
        for conn in connections[:5]:  # Top 5 connections
            connection_desc.append(
                f"- {conn.source_cluster} <-> {conn.target_cluster} (strength: {conn.connection_strength:.2f})"
            )

        # Format time span
        time_span = "N/A"
        if stats.time_range[0] and stats.time_range[1]:
            start_date = datetime.fromtimestamp(stats.time_range[0]).strftime("%Y-%m-%d")
            end_date = datetime.fromtimestamp(stats.time_range[1]).strftime("%Y-%m-%d")
            time_span = f"{start_date} ~ {end_date}"

        # Use enhanced prompt with metrics
        prompt = ENHANCED_PATTERN_DETECTION_PROMPT.format(
            total_conversations=stats.total_nodes,
            num_clusters=stats.total_clusters,
            time_span=time_span,
            cluster_summaries="\n".join(cluster_summaries),
            connections="\n".join(connection_desc) if connection_desc else "No strong connections identified",
            repetition_metrics=repetition_summary,
            # Future: add other metrics
            progression_metrics="(Not yet implemented)",
            gap_metrics="(Not yet implemented)",
            bridge_metrics="(Not yet implemented)",
        )

        try:
            response = self.llm_client.call_llm(
                system_prompt=f"You are an expert at analyzing conversation patterns. Base your analysis on the MEASURED METRICS provided. {self._lang_instruction}",
                user_prompt=prompt,
                temperature=0.7
            )

            data = self._parse_json_response(response)
            patterns_data = data.get("patterns", [])

            return [
                Pattern(
                    pattern_type=p.get("pattern_type", "unknown"),
                    description=p.get("description", ""),
                    evidence=p.get("evidence_summary", p.get("evidence", [])),  # Support both formats
                    significance=p.get("significance", "medium")
                )
                for p in patterns_data
            ]

        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return []

    # === Private Helper Methods ===

    def _generate_overview(
        self,
        stats,
        clusters: List
    ) -> OverviewSection:
        """Generate overview section."""
        # Format time span
        time_span = "N/A"
        if stats.time_range[0] and stats.time_range[1]:
            start_date = datetime.fromtimestamp(stats.time_range[0]).strftime("%Y-%m-%d")
            end_date = datetime.fromtimestamp(stats.time_range[1]).strftime("%Y-%m-%d")
            time_span = f"{start_date} ~ {end_date}"

        # Get top clusters
        sorted_clusters = sorted(clusters, key=lambda c: c.size, reverse=True)
        top_clusters_text = "\n".join(
            f"- {c.name}: {c.size} conversations"
            for c in sorted_clusters[:5]
        )

        prompt = OVERVIEW_PROMPT.format(
            total_conversations=stats.total_nodes,
            total_clusters=stats.total_clusters,
            time_span=time_span,
            top_clusters=top_clusters_text
        )

        try:
            response = self.llm_client.call_llm(
                system_prompt=f"You are an expert at analyzing conversation patterns and learning journeys. {self._lang_instruction}",
                user_prompt=prompt,
                temperature=0.7
            )

            data = self._parse_json_response(response)

            return OverviewSection(
                total_source_nodes=stats.total_nodes,
                time_span=time_span,
                primary_interests=data.get("primary_interests", [c.name for c in sorted_clusters[:3]]),
                conversation_style=data.get("conversation_style", "Exploratory"),
                most_active_period=data.get("most_active_period", "N/A"),
                summary_text=data.get("summary_text", f"A collection of {stats.total_nodes} conversations across {stats.total_clusters} topics.")
            )

        except Exception as e:
            logger.error(f"Overview generation failed: {e}")
            # Return basic overview
            return OverviewSection(
                total_source_nodes=stats.total_nodes,
                time_span=time_span,
                primary_interests=[c.name for c in sorted_clusters[:3]],
                conversation_style="Diverse",
                most_active_period="N/A",
                summary_text=f"A collection of {stats.total_nodes} conversations across {stats.total_clusters} topics."
            )

    def _generate_recommendations(
        self,
        overview: OverviewSection,
        cluster_analyses: List[ClusterAnalysis],
        patterns: List[Pattern]
    ) -> List[Recommendation]:
        """Generate actionable recommendations."""
        # Prepare summaries
        patterns_summary = "\n".join(
            f"- [{p.pattern_type}] {p.description}"
            for p in patterns[:5]
        )

        cluster_health = "\n".join(
            f"- {c.name}: {c.size} conversations, {c.recency} status"
            for c in cluster_analyses[:5]
        )

        prompt = RECOMMENDATION_PROMPT.format(
            primary_interests=", ".join(overview.primary_interests),
            conversation_style=overview.conversation_style,
            total_conversations=overview.total_source_nodes,
            patterns_summary=patterns_summary if patterns_summary else "No significant patterns detected",
            cluster_health=cluster_health
        )

        try:
            response = self.llm_client.call_llm(
                system_prompt=f"You are an expert learning advisor analyzing conversation patterns. {self._lang_instruction}",
                user_prompt=prompt,
                temperature=0.7
            )

            data = self._parse_json_response(response)
            recs_data = data.get("recommendations", [])

            return [
                Recommendation(
                    type=r.get("type", "explore"),
                    title=r.get("title", ""),
                    description=r.get("description", ""),
                    related_nodes=[],  # Could be populated with evidence
                    priority=r.get("priority", "medium")
                )
                for r in recs_data[:5]  # Limit to 5
            ]

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []

    def _compute_cluster_density(self, cluster_id: str) -> float:
        """Compute intra-cluster edge density."""
        cluster_with_nodes = self.graph_loader.get_cluster_with_nodes(cluster_id)
        if not cluster_with_nodes or cluster_with_nodes.size <= 1:
            return 0.0

        # Maximum possible edges in cluster
        n = cluster_with_nodes.size
        max_edges = n * (n - 1) / 2

        if max_edges == 0:
            return 0.0

        # Actual internal edges
        actual_edges = cluster_with_nodes.internal_edge_count

        return min(1.0, actual_edges / max_edges)

    def _compute_cluster_centrality(self, cluster_id: str) -> float:
        """Compute how connected this cluster is to others."""
        # Get all edges from graph
        all_edges = self.graph_loader.graph_store.get_all_edges()
        cluster_nodes = self.graph_loader.graph_store.get_nodes_by_cluster(cluster_id)
        cluster_node_ids = {str(n.get("id")) for n in cluster_nodes}

        if not cluster_node_ids:
            return 0.0

        # Count inter-cluster edges
        inter_cluster_edges = 0
        for edge in all_edges:
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))

            # Check if edge connects this cluster to another
            source_in_cluster = source in cluster_node_ids
            target_in_cluster = target in cluster_node_ids

            if source_in_cluster != target_in_cluster:  # XOR - one in, one out
                inter_cluster_edges += 1

        # Normalize by cluster size
        if len(cluster_node_ids) == 0:
            return 0.0

        # Heuristic: centrality based on inter-cluster edges per node
        centrality = min(1.0, inter_cluster_edges / (len(cluster_node_ids) * 2))
        return centrality

    def _compute_recency(self, cluster_id: str) -> str:
        """Determine if cluster is active, dormant, or new based on update_time.

        Categories:
        - "active": updated within last 7 days
        - "recent": updated within last 30 days
        - "dormant": updated within last 90 days
        - "stale": not updated in 90+ days
        - "new": created within last 7 days (regardless of updates)
        - "unknown": no time data available
        """
        import time

        cluster_nodes = self.graph_loader.graph_store.get_nodes_by_cluster(cluster_id)

        update_times = [n.get("update_time") for n in cluster_nodes if n.get("update_time")]
        create_times = [n.get("create_time") for n in cluster_nodes if n.get("create_time")]

        if not update_times and not create_times:
            return "unknown"

        now = int(time.time())

        # Check if cluster is "new" (most recent creation within 7 days)
        if create_times:
            latest_create = max(create_times)
            days_since_creation = (now - latest_create) / 86400
            if days_since_creation <= 7:
                return "new"

        # Determine recency based on most recent update
        if update_times:
            latest_update = max(update_times)
            days_since_update = (now - latest_update) / 86400

            if days_since_update <= 7:
                return "active"
            elif days_since_update <= 30:
                return "recent"
            elif days_since_update <= 90:
                return "dormant"
            else:
                return "stale"

        # Fallback to create_time if no update_time
        if create_times:
            latest_create = max(create_times)
            days_since_creation = (now - latest_create) / 86400

            if days_since_creation <= 30:
                return "recent"
            elif days_since_creation <= 90:
                return "dormant"
            else:
                return "stale"

        return "unknown"

    def _compute_cluster_time_stats(self, cluster_id: str) -> dict:
        """Compute detailed time statistics for a cluster.

        Returns:
            Dictionary with time statistics including creation/update times,
            time span, and recency information.
        """
        import time

        cluster_nodes = self.graph_loader.graph_store.get_nodes_by_cluster(cluster_id)

        create_times = [n.get("create_time") for n in cluster_nodes if n.get("create_time")]
        update_times = [n.get("update_time") for n in cluster_nodes if n.get("update_time")]

        stats = {
            "total_nodes": len(cluster_nodes),
            "nodes_with_time_data": len(create_times),
        }

        if create_times:
            stats["earliest_created"] = datetime.fromtimestamp(min(create_times)).isoformat()
            stats["latest_created"] = datetime.fromtimestamp(max(create_times)).isoformat()
            stats["creation_span_days"] = (max(create_times) - min(create_times)) / 86400

        if update_times:
            stats["latest_updated"] = datetime.fromtimestamp(max(update_times)).isoformat()
            now = int(time.time())
            stats["days_since_last_update"] = (now - max(update_times)) / 86400

        return stats

    def _detect_temporal_patterns(self) -> List[dict]:
        """Detect temporal patterns across the entire graph.

        Focuses on:
        - Progression: Learning journey from basic to advanced topics
        - Recent surge: Sudden increase in activity on a topic
        - Revisited: Returning to a topic after a long gap
        - Cross-cluster progression: Moving between related topics over time
        """
        import time
        from collections import defaultdict

        patterns = []
        now = int(time.time())

        # Get all nodes with time data, sorted chronologically
        all_nodes = self.graph_loader.get_all_nodes()
        nodes_with_time = [n for n in all_nodes if n.create_time]
        nodes_with_time.sort(key=lambda n: n.create_time)

        if len(nodes_with_time) < 5:
            return patterns

        # English comment.
        # English comment.
        cluster_sequence = []
        for node in nodes_with_time:
            cluster_id = node.cluster_id
            if not cluster_sequence or cluster_sequence[-1] != cluster_id:
                cluster_sequence.append(cluster_id)

        # English comment.
        cluster_revisit_counts = defaultdict(int)
        for i, cluster_id in enumerate(cluster_sequence):
            if cluster_id in cluster_sequence[:i]:
                cluster_revisit_counts[cluster_id] += 1

        for cluster_id, revisit_count in cluster_revisit_counts.items():
            if revisit_count >= 2:
                cluster = self.graph_loader.get_cluster(cluster_id)
                cluster_name = cluster.name if cluster else cluster_id
                patterns.append({
                    "pattern_type": "revisited_topic",
                    "description": f"'{cluster_name}' text {revisit_count + 1}text text text (text text)",
                    "evidence": [],
                    "significance": "high" if revisit_count >= 3 else "medium"
                })

        # English comment.
        # English comment.
        clusters = self.graph_loader.get_all_clusters()

        for cluster in clusters:
            cluster_nodes = [n for n in nodes_with_time if n.cluster_id == cluster.id]
            if len(cluster_nodes) < 3:
                continue

            # English comment.
            early_keywords = set()
            late_keywords = set()

            # English comment.
            split_idx = len(cluster_nodes) // 3
            if split_idx < 1:
                continue

            for node in cluster_nodes[:split_idx]:
                early_keywords.update(node.get_keyword_terms(top_n=5))
            for node in cluster_nodes[-split_idx:]:
                late_keywords.update(node.get_keyword_terms(top_n=5))

            # English comment.
            new_keywords = late_keywords - early_keywords

            # English comment.
            if len(new_keywords) >= 3 and len(late_keywords) > len(early_keywords):
                patterns.append({
                    "pattern_type": "progression",
                    "description": f"'{cluster.name}' text text text text (text text: {', '.join(list(new_keywords)[:3])})",
                    "evidence": [cluster_nodes[0].orig_id, cluster_nodes[-1].orig_id],
                    "significance": "high"
                })

        # English comment.
        recent_threshold = now - (14 * 86400)  # English comment.
        recent_nodes = [n for n in nodes_with_time if n.create_time >= recent_threshold]

        if len(recent_nodes) >= 5:
            # English comment.
            recent_cluster_counts = defaultdict(int)
            for node in recent_nodes:
                recent_cluster_counts[node.cluster_id] += 1

            top_recent_cluster = max(recent_cluster_counts, key=recent_cluster_counts.get)
            top_count = recent_cluster_counts[top_recent_cluster]

            if top_count >= 3:
                cluster = self.graph_loader.get_cluster(top_recent_cluster)
                cluster_name = cluster.name if cluster else top_recent_cluster
                patterns.append({
                    "pattern_type": "recent_surge",
                    "description": f"text 2text '{cluster_name}' text textin progress ({top_count}text text)",
                    "evidence": [n.orig_id for n in recent_nodes if n.cluster_id == top_recent_cluster][:3],
                    "significance": "high"
                })

        # English comment.
        # English comment.
        transition_counts = defaultdict(int)
        for i in range(len(cluster_sequence) - 1):
            pair = (cluster_sequence[i], cluster_sequence[i + 1])
            if pair[0] != pair[1]:  # English comment.
                transition_counts[pair] += 1

        for (from_cluster, to_cluster), count in transition_counts.items():
            if count >= 2:
                from_c = self.graph_loader.get_cluster(from_cluster)
                to_c = self.graph_loader.get_cluster(to_cluster)
                from_name = from_c.name if from_c else from_cluster
                to_name = to_c.name if to_c else to_cluster
                patterns.append({
                    "pattern_type": "learning_path",
                    "description": f"'{from_name}' → '{to_name}' text text text ({count}text)",
                    "evidence": [],
                    "significance": "medium"
                })

        # Sort by significance and limit
        significance_order = {"high": 0, "medium": 1, "low": 2}
        patterns.sort(key=lambda p: significance_order.get(p["significance"], 99))

        return patterns[:10]  # Top 10 patterns

    def _find_cluster_connections(self, clusters: List) -> List[ClusterConnection]:
        """Find connections between clusters."""
        connections = []
        all_edges = self.graph_loader.graph_store.get_all_edges()

        # Build cluster pair edge counts
        cluster_pairs = {}
        for edge in all_edges:
            if not edge.get("is_intra_cluster", False):
                # Get source and target nodes
                source_node = self.graph_loader.graph_store.get_node(str(edge.get("source", "")))
                target_node = self.graph_loader.graph_store.get_node(str(edge.get("target", "")))

                if source_node and target_node:
                    source_cluster = source_node.get("cluster_id")
                    target_cluster = target_node.get("cluster_id")

                    if source_cluster and target_cluster and source_cluster != target_cluster:
                        # Normalize pair order
                        pair = tuple(sorted([source_cluster, target_cluster]))
                        cluster_pairs[pair] = cluster_pairs.get(pair, 0) + 1

        # Create connections
        for (c1, c2), count in cluster_pairs.items():
            # Find cluster names
            cluster1 = next((c for c in clusters if c.id == c1), None)
            cluster2 = next((c for c in clusters if c.id == c2), None)

            if cluster1 and cluster2:
                # Simple strength calculation
                strength = min(1.0, count / 5.0)  # Normalize, 5+ edges = strong

                connections.append(ClusterConnection(
                    source_cluster=cluster1.name,
                    target_cluster=cluster2.name,
                    connection_strength=strength,
                    bridge_keywords=[],  # Could be computed from shared keywords
                    description=f"{count} connections between these topics"
                ))

        # Sort by strength
        connections.sort(key=lambda c: c.connection_strength, reverse=True)
        return connections[:10]  # Top 10 connections

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response, handling common issues."""
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse: {response}")
            return {}
