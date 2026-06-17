"""Prompts for graph summarization and analysis."""

# ---------------------------------------------------------------------------
# Language instruction helpers
# ---------------------------------------------------------------------------

LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "ko": "Respond in Korean (Respond in Korean).",
    "en": "Respond in English.",
    "zh": "请用中文（简体）回答。",
}


def get_language_instruction(language: str) -> str:
    """Return the language instruction string to append to system prompts.

    Args:
        language: ISO 639-1 language code (e.g. ``"ko"``, ``"en"``, ``"zh"``).
            Falls back to English for unknown codes.

    Returns:
        A short instruction sentence understood by LLMs.
    """
    return LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])


# ---------------------------------------------------------------------------

OVERVIEW_PROMPT = """You are analyzing a knowledge graph built from a user's LLM conversation history.

## Graph Statistics
Total conversations: {total_conversations}
Total clusters: {total_clusters}
Time span: {time_span}
Top clusters by size:
{top_clusters}

## Task
Generate a comprehensive overview of this user's conversation patterns and interests.

## Output Format (JSON)
{{
  "primary_interests": ["interest 1", "interest 2", "interest 3"],
  "conversation_style": "A brief characterization (e.g., 'technical deep-dive', 'exploratory', 'problem-solving focused')",
  "most_active_period": "Description of when most conversations occur (or 'N/A' if no timestamp data)",
  "summary_text": "2-3 sentence natural language summary of the user's conversation patterns and learning journey"
}}

## Guidelines
- Identify the 3-5 most prominent topics/interests based on cluster sizes and names
- Characterize the conversation style (e.g., deep technical dives, exploratory, tutorial-seeking, etc.)
- Write the summary in a natural, insightful tone
- Support multilingual content (Korean, English, Chinese)
- Be specific about observed patterns

Output only valid JSON."""

CLUSTER_ANALYSIS_PROMPT = """You are analyzing a specific cluster in a conversation knowledge graph.

## Cluster Information
Name: {cluster_name}
Description: {cluster_description}
Size: {cluster_size} conversations
Internal edge density: {density:.2f}
Centrality: {centrality:.2f}
Recency: {recency}

## Top Keywords
{top_keywords}

## Sample Conversation Topics
{sample_topics}

## Task
Provide a detailed analysis of this cluster, identifying themes, patterns, and insights.

## Output Format (JSON)
{{
  "key_themes": ["theme 1", "theme 2", "theme 3"],
  "common_question_types": ["type 1", "type 2"],
  "insight_text": "2-3 sentence analysis of what this cluster reveals about the user's interests and learning patterns",
  "notable_patterns": "Any interesting patterns you notice (e.g., progression from basics to advanced topics)"
}}

## Guidelines
- Identify 3-5 key themes within the cluster
- Categorize common question types (e.g., "debugging", "concept explanation", "comparison", "implementation")
- Provide actionable insights about the user's learning journey
- Note any progression or evolution in complexity
- Be specific and avoid generic observations

Output only valid JSON."""

PATTERN_DETECTION_PROMPT = """You are analyzing patterns across a conversation knowledge graph.

## Graph Overview
Total conversations: {total_conversations}
Clusters: {num_clusters}
Time span: {time_span}

## Cluster Summaries
{cluster_summaries}

## Inter-cluster Connections
{connections}

## Task
Identify meaningful patterns across the entire conversation history.

## Pattern Types
1. **Repetition**: Topics or questions that recur multiple times
2. **Progression**: Evolution from basic to advanced topics
3. **Gap**: Topics that were explored once but never revisited
4. **Bridge**: Topics that connect multiple areas of interest

## Output Format (JSON)
{{
  "patterns": [
    {{
      "pattern_type": "repetition|progression|gap|bridge",
      "description": "Clear description of the pattern",
      "evidence": ["conv_id_1", "conv_id_2"],
      "significance": "high|medium|low"
    }}
  ]
}}

## Guidelines
- Focus on significant patterns (not trivial observations)
- Provide specific evidence (conversation IDs when possible)
- Prioritize patterns that reveal learning journey or knowledge gaps
- Limit to 3-5 most important patterns
- Be concise but specific

Output only valid JSON."""

ENHANCED_PATTERN_DETECTION_PROMPT = """You are analyzing patterns in a conversation knowledge graph.

## Graph Overview
- Total conversations: {total_conversations}
- Clusters: {num_clusters}
- Time span: {time_span}

## Cluster Summaries
{cluster_summaries}

## Inter-cluster Connections
{connections}

---

# MEASURED METRICS (Use these as evidence)

{repetition_metrics}

{progression_metrics}

{gap_metrics}

{bridge_metrics}

---

## Task
Based on the **MEASURED METRICS** above, identify and describe meaningful patterns.

## Important Guidelines
1. **Use only the evidence provided** - Do not invent conversation IDs or metrics
2. **Reference specific numbers** - e.g., "pytorch appears in 8 conversations"
3. **Cite actual conversation pairs** - e.g., "conv_5 and conv_12 share 89% similarity"
4. **Base significance on measured values** - High similarity (>85%) or high frequency (>5) = high significance
5. **Support multilingual content** - Handle Korean, English, and Chinese keywords appropriately

## Pattern Types
1. **Repetition**: Topics or questions that recur multiple times (use repetition_metrics)
2. **Progression**: Evolution from basic to advanced topics (use progression_metrics when available)
3. **Gap**: Topics that were explored once but never revisited (use gap_metrics when available)
4. **Bridge**: Topics that connect multiple areas of interest (use bridge_metrics or cross-cluster keywords)

## Output Format (JSON)
{{
  "patterns": [
    {{
      "pattern_type": "repetition|progression|gap|bridge",
      "description": "Description citing specific metrics",
      "evidence_summary": "Key numbers: X conversations, Y% similarity, Z shared keywords",
      "supporting_node_ids": ["actual_conv_ids_from_metrics"],
      "significance": "high|medium|low"
    }}
  ]
}}

Focus on patterns with strong measured evidence. It's better to report fewer patterns with solid evidence than many patterns with weak support.

Output only valid JSON."""

RECOMMENDATION_PROMPT = """You are generating actionable recommendations based on conversation graph analysis.

## User Profile
Primary interests: {primary_interests}
Conversation style: {conversation_style}
Total conversations: {total_conversations}

## Identified Patterns
{patterns_summary}

## Cluster Health
{cluster_health}

## Task
Generate 3-5 actionable recommendations to help the user maximize their learning.

## Recommendation Types
1. **Consolidate**: Similar questions/topics that should be consolidated into a guide
2. **Explore**: Related topics the user might find interesting
3. **Review**: Important topics that haven't been revisited in a while
4. **Connect**: Opportunities to connect knowledge across domains

## Output Format (JSON)
{{
  "recommendations": [
    {{
      "type": "consolidate|explore|review|connect",
      "title": "Short, actionable title",
      "description": "1-2 sentence explanation of why this is recommended",
      "priority": "high|medium|low"
    }}
  ]
}}

## Guidelines
- Prioritize recommendations that provide clear value
- Be specific (mention actual topics/clusters)
- Consider both depth (mastery) and breadth (exploration)
- Identify repeated pain points or gaps
- Limit to 5 recommendations maximum

Output only valid JSON."""
