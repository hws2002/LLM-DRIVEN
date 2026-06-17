"""Schema-first extraction prompts for nodes and edges."""
import json
from typing import Dict

from microscope.utils.io_utils import load_ontology_schema

_STRIP_KEYS = ("_comment", "meta", "merge_policy", "extraction_output")

def _build_system_prompt(schema: Dict) -> str:
    schema = {k: v for k, v in schema.items() if k not in _STRIP_KEYS}
    schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
    return _PROMPT_TEMPLATE.format(schema_str=schema_str)


def get_system_prompt_with_schema_dict(schema: Dict) -> str:
    """English documentation."""
    return _build_system_prompt(schema)


def get_system_prompt_with_schema(schema_path: str) -> str:
    """English documentation."""
    schema = load_ontology_schema(schema_path)
    return _build_system_prompt(schema)


_PROMPT_TEMPLATE = """\
You are a precision-focused Knowledge Graph Extractor for a specialized domain.
Your task is to extract entities (nodes) and relations (edges) from the provided text
strictly adhering to a predefined **Provided Schema**.
Remember your role: return only structured JSON that matches the schema.

You will receive a schema definition below. You must dynamically adapt your extraction logic based on this schema.

* **Node/Edge Types:** usage is restricted strictly to the `node_types` and `edge_types` lists defined in the schema. **NO hallucinations.**
* **Field Semantics:** In `node_fields` and `edge_fields`, the key is the field name, and the value describes the **REQUIREMENT**.
    * *Example:* If the schema says `"evidence": "verbatim quote"`, you MUST provide a verbatim quote.
    * *Example:* If the schema says `"summary": "short text"`, you MUST provide a short text.
    * **Instruction:** Read the description string for each field in the schema and execute that specific instruction precisely.

CRITICAL INSTRUCTION - SCHEMA COMPLIANCE:
1. You must ONLY use the Node Types provided in the schema. Do not invent new types.
2. You must ONLY use the Relation Types provided in the schema. If a relationship doesn't fit the schema, ignore it or fit it to the closest allowed relation.
3. Entities must be atomic and consistent.
4. **Edge Type Constraints:** Some edge types define `typical_pairs` and `forbidden_pairs`.
   - `typical_pairs`: a list of [source_type, target_type] pairs. The `start` node type and `target` node type of the edge MUST match one of these pairs.
   - `forbidden_pairs`: a list of [source_type, target_type] pairs (where "any" means any type). You MUST NOT create an edge whose start/target types match any forbidden pair.
   - If an edge has both, `forbidden_pairs` takes priority.
5. **Weak & Indirect Relations:** Do NOT omit edges just because the relationship is weak, indirect, or uncertain. Include them with a low confidence value (0.3–0.5). Aim for at least 1.5 edges per node on average. It is better to include a weak edge than to miss a real relationship.
6. **Language Consistency:** All node names MUST reflect the language as it appears in the source document. Do NOT translate terms — preserve each term in the language it appears in the source. If the document is in Korean, use Korean. If English, use English. If Chinese, use Chinese. If the document mixes languages, preserve the mixed-language style as-is.
7. **Node-Edge Consistency:** Every node referenced in an edge's 'start' or 'target' MUST exist in the nodes list. Do NOT create edges that reference nodes you have not extracted.

### 2. STRICT OUTPUT FORMAT
Return a SINGLE JSON object containing "nodes" and "edges" lists.
Output the JSON object inside a ```json code block and nothing else.
* **Nodes:** Must only contain fields defined in `node_fields`.
* **Edges:** Must only contain fields defined in `edge_fields`.
* **Consistency:** The `start` and `target` in edges must strictly match the `name` of a node in the nodes list.
* **Chunk Tracking:** Set source_chunk_id using the Chunk numbers above.

### **PROVIDED SCHEMA :**
{schema_str}
"""


USER_WANTED_PROMPT = """
Extract all key entities and relationships from the text with balanced coverage.
"""
