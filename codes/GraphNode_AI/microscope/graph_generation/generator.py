import base64
import tiktoken
import logging
import time
import concurrent.futures
from shared.api_provider import ApiProvider
from ..utils.io_utils import extract_json_from_text, load_ontology_schema
from shared.token_usage import TokenUsageTracker
from ..prompts.prompt_factory import PromptFactory
from langchain_core.documents import Document
from typing import List, Tuple, Dict, Any

_SLIDE_GRAPH_SYSTEM_PROMPT = """\
You are a presentation structure analyzer.
You will receive slides (as images) from a single presentation.

Your task:
1. Create exactly ONE node per slide.
2. Extract meaningful edges between slides based on their relationships.

Node format:
  name        : "slide_{N}"  (N = 1-based slide number, e.g. "slide_1")
  label       : short topic title (1-5 words, in the slide's language)
  type        : "Slide"
  description : one-sentence summary of the slide content

Edge types (use only these):
  leads_to       — natural topical flow to the next subject
  elaborates     — expands on a concept introduced in another slide
  contrasts_with — presents opposing or contrasting content
  supports       — provides evidence or examples for another slide
  summarizes     — summarizes content from previous slides
  introduces     — introduces a concept developed in a later slide

Rules:
- node name MUST follow the pattern "slide_{N}" exactly.
- Only create edges where a meaningful relationship exists (sequential order alone is not enough for leads_to).
- Preserve the language of the slide content in label and description.
- Every node referenced in an edge MUST exist in the nodes list.

Return a single JSON object inside a ```json code block:
{
  "nodes": [
    {"name": "slide_1", "label": "...", "type": "Slide", "description": "..."},
    ...
  ],
  "edges": [
    {"start": "slide_1", "target": "slide_2", "type": "leads_to", "description": "..."},
    ...
  ]
}
"""

logger = logging.getLogger("GraphGenerator")

class GraphGenerator:
    def __init__(self, api_provider: ApiProvider, user_id: str | None = None,
                 ontology_schema: dict | None = None, batch_max_tokens: int = 6000,
                 modifier: str | None = None):
        self.api_provider = api_provider
        self.model = api_provider.model
        self.ontology_schema = ontology_schema or {}
        self.prompt_factory = PromptFactory(modifier=modifier, schema=self.ontology_schema)
        self.batch_max_tokens = batch_max_tokens
        self.chunks = None
        self.tracker = TokenUsageTracker(
            model_name=api_provider.model,
            provider_name=api_provider.provider or "unknown",
        )

    def _load_forbidden_pairs(self) -> Dict[str, List[List[str]]]:
        """English documentation."""
        return {
            edge["name"]: edge["forbidden_pairs"]
            for edge in self.ontology_schema.get("edge_types", [])
            if edge.get("forbidden_pairs")
        }

    def _filter_invalid_edge_types(self, edges: List[Dict]) -> List[Dict]:
        """English documentation."""
        valid_types = {e["name"] for e in self.ontology_schema.get("edge_types", [])}
        return [e for e in edges if e.get("type") in valid_types]

    def _filter_forbidden_edges(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        forbidden_pairs: Dict[str, List[List[str]]],
    ) -> List[Dict]:
        """English documentation."""
        if not forbidden_pairs:
            return edges

        node_type_map: Dict[str, List[str]] = {}
        for node in nodes:
            name = node.get("name", "")
            ntype = node.get("type", "")
            if name:
                node_type_map[name] = ntype if isinstance(ntype, list) else ([ntype] if ntype else [])

        filtered = []
        for edge in edges:
            etype = edge.get("type", "")
            pairs = forbidden_pairs.get(etype, [])
            if not pairs:
                filtered.append(edge)
                continue

            start_types = node_type_map.get(edge.get("start", ""), ["unknown"])
            target_types = node_type_map.get(edge.get("target", ""), ["unknown"])

            violated = any(
                (fp[0] == "any" or fp[0] == st) and (fp[1] == "any" or fp[1] == tt)
                for fp in pairs
                for st in start_types
                for tt in target_types
            )

            if violated:
                logger.info(
                    "  [FILTERED] %s -[%s]-> %s (forbidden_pairs violation)",
                    edge.get("start", ""), etype, edge.get("target", ""),
                )
            else:
                filtered.append(edge)

        return filtered

    def _filter_dangling_edges(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """English documentation."""
        node_names = {n.get("name", "") for n in nodes if n.get("name")}
        filtered = []
        for edge in edges:
            start = edge.get("start", "")
            if start == edge.get("target", ""):
                continue
            target = edge.get("target", "")
            if start in node_names and target in node_names:
                filtered.append(edge)
            else:
                logger.info(
                    "  [FILTERED] %s -[%s]-> %s (dangling edge — node not found)",
                    start, edge.get("type", ""), target,
                )
        return filtered

    def _invoke_llm(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.api_provider.chat_completion_text(
            messages=messages,
        )

    def _build_user_prompt(self, batch_chunks_index_range, user_wanted_prompt) -> str:
        parts = []
        for idx in batch_chunks_index_range:
            parts.append(f"[Chunk {idx}/{len(self.chunks)}]\n{self.chunks[idx].page_content}")
        parts.append(
            "Extract nodes/edges JSON for the content above.\n" +
            user_wanted_prompt
        )
        return "\n\n".join(parts)

    def extract_entity_relation_from_chunks(self,
        chunks: List[Document],
        verbose: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """English documentation."""

        if verbose:
            logger.info("Extracting graph from chunks...")

        entity_relations: List[Dict[str, Any]] = []
        raw_llm_outputs: List[str] = []
        self.chunks = chunks

        try:
            enc = tiktoken.encoding_for_model(self.model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")

        schema_extraction_system_prompt = self.prompt_factory.get_prompt("extraction_system")
        forbidden_pairs = self._load_forbidden_pairs()
        logger.info("Encoding chunk content...")
        num_system_tokens = len(enc.encode(schema_extraction_system_prompt))
        fixed_suffix_tokens = len(enc.encode("Extract nodes/edges JSON for the content above."))
        num_chunk_tokens = [len(enc.encode(chunk.page_content)) for chunk in chunks]
        num_batch_tokens = num_system_tokens + fixed_suffix_tokens
        batch_indexes = [0]
        user_tokens = []
        for idx, cost in enumerate(num_chunk_tokens):
            num_batch_tokens += cost
            if idx == len(num_chunk_tokens) - 1:
                batch_indexes.append(idx+1)
                user_tokens.append(num_batch_tokens - num_system_tokens)
                break
            if num_batch_tokens + num_chunk_tokens[idx+1] + 100 > self.batch_max_tokens:
                batch_indexes.append(idx+1)
                user_tokens.append(num_batch_tokens - num_system_tokens)
                num_batch_tokens = num_system_tokens + fixed_suffix_tokens
        logger.info("LLMtext text text text: %s", len(batch_indexes) - 1)
        for b_idx in range(len(batch_indexes)-1):
            user_prompt = self._build_user_prompt(range(batch_indexes[b_idx], batch_indexes[b_idx+1]), self.prompt_factory.get_prompt("extraction_user_wanted"))

            logger.info("LLMtext textin progress: [%s/%s]", b_idx+1, len(batch_indexes) - 1)
            _t0 = time.time()
            raw = self._invoke_llm(schema_extraction_system_prompt, user_prompt)
            logger.info("LLM completed: [%s/%s] (%.1fs)", b_idx+1, len(batch_indexes) - 1, time.time() - _t0)

            self.tracker.record_call(
                stage="extraction",
                system_prompt=schema_extraction_system_prompt,
                user_prompt=user_prompt,
                response=raw,
                max_tokens=self.api_provider.max_tokens,
                temperature=self.api_provider.temperature,
                metadata={
                    "batch_index": b_idx + 1,
                    "batch_start": batch_indexes[b_idx],
                    "batch_end": batch_indexes[b_idx+1],
                },
            )

            raw_llm_outputs.append(raw)
            logger.info("Raw LLM output: %s", raw[:200])

            graph_data = extract_json_from_text(raw)
            if not isinstance(graph_data, dict):
                graph_data = {"nodes": [], "edges": []}

            # English comment.
            for edge in graph_data.get("edges", []):
                if "start" not in edge and "source" in edge:
                    edge["start"] = edge.pop("source")

            nodes = graph_data.get("nodes", [])
            for node in nodes:
                t = node.get("type")
                if isinstance(t, list) and len(t) == 1 and isinstance(t[0], list):
                    node["type"] = t[0]  # [['Event']] → ['Event']
            edges = self._filter_invalid_edge_types(graph_data.get("edges", []))
            edges = self._filter_forbidden_edges(nodes, edges, forbidden_pairs)
            edges = self._filter_dangling_edges(nodes, edges)
            graph_data["edges"] = edges
            logger.info(
                "text [%s/%s] text text — text %dtext, text %dtext",
                b_idx + 1, len(batch_indexes) - 1, len(nodes), len(edges),
            )
            for n in nodes:
                logger.info("  NODE  %-20s [%s]", n.get("name", ""), n.get("type", ""))
            for e in edges:
                logger.info("  EDGE  %s -[%s]-> %s", e.get("start", ""), e.get("type", ""), e.get("target", ""))

            entity_relations.append(graph_data)

        return entity_relations, raw_llm_outputs

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.strip().lower().split())

    def _collect_entity_names_by_type(
        self,
        entity_relations: List[Dict[str, List[dict]]],
        existing_nodes: List[Dict] | None = None,
    ) -> Dict[str, List[str]]:
        """English documentation."""
        type_to_names: Dict[str, List[str]] = {}
        seen_per_type: Dict[str, set] = {}

        def add(name: str, ntype: str) -> None:
            if not name or not ntype:
                return
            norm = self._normalize_name(name)
            if ntype not in type_to_names:
                type_to_names[ntype] = []
                seen_per_type[ntype] = set()
            if norm not in seen_per_type[ntype]:
                seen_per_type[ntype].add(norm)
                type_to_names[ntype].append(name)

        for item in entity_relations:
            for node in item.get("nodes", []):
                ntypes = node.get("type", "")
                if isinstance(ntypes, list):
                    for t in ntypes:
                        add(node.get("name", ""), t)
                else:
                    add(node.get("name", ""), ntypes)

        if existing_nodes:
            for node in existing_nodes:
                ntypes = node.get("types") or []
                if isinstance(ntypes, str):
                    ntypes = [ntypes]
                for ntype in ntypes:
                    add(node.get("name", ""), ntype)

        return type_to_names

    def standardize_extracted_graph(self,
        extracted_entity_relations: List[Dict[str, List[dict]]],
        existing_nodes: List[Dict] | None,
    ) -> Tuple[List[Dict[str, List[dict]]], Dict[str, str]]:
        """English documentation."""
        type_to_names = self._collect_entity_names_by_type(extracted_entity_relations, existing_nodes)
        if not type_to_names:
            return extracted_entity_relations, {}

        combined_mapping: Dict[str, str] = {}
        system_prompt = self.prompt_factory.get_prompt("standardization_system")

        def _standardize_type(ntype: str, names: List[str]) -> tuple:
            entity_names_sorted = "\n".join(sorted(names, key=str.lower))
            user_prompt = self.prompt_factory.get_prompt("standardization_user", entity_names_sorted)
            logger.info("Standardizing type '%s' (%d entities)...", ntype, len(names))
            raw = self._invoke_llm(system_prompt, user_prompt)
            mapping_json = extract_json_from_text(raw)
            if not isinstance(mapping_json, dict):
                return {}, system_prompt, user_prompt, raw
            result: Dict[str, str] = {}
            for standard, variants in mapping_json.items():
                if not isinstance(variants, list):
                    continue
                result[self._normalize_name(standard)] = standard
                for v in variants:
                    if isinstance(v, str):
                        result[self._normalize_name(v)] = standard
            return result, system_prompt, user_prompt, raw

        eligible = {ntype: names for ntype, names in type_to_names.items() if len(names) >= 2}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(_standardize_type, ntype, names): ntype for ntype, names in eligible.items()}
            for future in concurrent.futures.as_completed(futures):
                mapping, sys_p, usr_p, raw = future.result()
                combined_mapping.update(mapping)
                self.tracker.record_call(
                    stage="standardization",
                    system_prompt=sys_p,
                    user_prompt=usr_p,
                    response=raw,
                    max_tokens=self.api_provider.max_tokens,
                    temperature=self.api_provider.temperature,
                )

        for item in extracted_entity_relations:
            for node in item.get("nodes", []):
                name = node.get("name", "")
                if name:
                    node["name"] = combined_mapping.get(self._normalize_name(name), name)
            for edge in item.get("edges", []):
                src = edge.get("start", "")
                legacy_source = False
                if not src and edge.get("source") and edge.get("target"):
                    src = edge.get("source", "")
                    legacy_source = True
                tgt = edge.get("target", "")
                if src:
                    edge["start"] = combined_mapping.get(self._normalize_name(src), src)
                    if legacy_source:
                        edge.pop("source", None)
                if tgt:
                    edge["target"] = combined_mapping.get(self._normalize_name(tgt), tgt)

        return extracted_entity_relations, combined_mapping

    def extract_entity_relation_from_images(
        self,
        slide_images: List[bytes],
        batch_size: int = 10,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """English documentation."""
        all_results: List[Dict[str, Any]] = []
        raw_outputs: List[str] = []
        total = len(slide_images)
        total_batches = (total + batch_size - 1) // batch_size

        _VALID_SLIDE_EDGE_TYPES = {
            "leads_to", "elaborates", "contrasts_with",
            "supports", "summarizes", "introduces",
        }

        for batch_start in range(0, total, batch_size):
            batch = slide_images[batch_start: batch_start + batch_size]
            batch_end = batch_start + len(batch)
            b_idx = batch_start // batch_size + 1
            logger.info("text text processing: [%d/%d] (slides %d-%d)",
                        b_idx, total_batches, batch_start + 1, batch_end)

            content: List[dict] = []
            for i, img_bytes in enumerate(batch):
                slide_num = batch_start + i + 1
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                content.append({"type": "text", "text": f"[Slide {slide_num}]"})
                content.append({"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{b64}"}})

            content.append({
                "type": "text",
                "text": f"Above are slides {batch_start + 1} to {batch_end}. Extract the slide graph as specified.",
            })

            messages = [
                {"role": "system", "content": _SLIDE_GRAPH_SYSTEM_PROMPT},
                {"role": "user",   "content": content},
            ]

            _t0 = time.time()
            raw = self.api_provider.chat_completion_text(messages=messages)
            logger.info("LLM completed: [%d/%d] (%.1fs)", b_idx, total_batches, time.time() - _t0)
            raw_outputs.append(raw)

            graph_data = extract_json_from_text(raw)
            if not isinstance(graph_data, dict):
                graph_data = {"nodes": [], "edges": []}

            for edge in graph_data.get("edges", []):
                if "start" not in edge and "source" in edge:
                    edge["start"] = edge.pop("source")

            nodes = graph_data.get("nodes", [])
            edges = [e for e in graph_data.get("edges", []) if e.get("type") in _VALID_SLIDE_EDGE_TYPES]
            edges = self._filter_dangling_edges(nodes, edges)
            graph_data["edges"] = edges

            logger.info("text [%d/%d] — text %dtext, text %dtext",
                        b_idx, total_batches, len(nodes), len(edges))
            for n in nodes:
                logger.info("  NODE  %-20s [%s]", n.get("name", ""), n.get("type", ""))
            for e in edges:
                logger.info("  EDGE  %s -[%s]-> %s",
                            e.get("start", ""), e.get("type", ""), e.get("target", ""))

            self.tracker.record_call(
                stage="slide_extraction",
                system_prompt=_SLIDE_GRAPH_SYSTEM_PROMPT,
                user_prompt=f"slides {batch_start + 1}-{batch_end}",
                response=raw,
                max_tokens=self.api_provider.max_tokens,
                temperature=self.api_provider.temperature,
                metadata={"batch_start": batch_start + 1, "batch_end": batch_end},
            )

            all_results.append(graph_data)

        return all_results, raw_outputs
