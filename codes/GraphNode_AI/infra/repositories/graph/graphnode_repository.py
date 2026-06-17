"""Unified repository entry point — GraphNodeDBHandler only.

Neo4jHandler, VectorDBHandler, ConversationNodeStore have been moved to:
  infra/repositories/neo4j/handler.py
  infra/repositories/vectordb/chunks_store.py
  infra/repositories/vectordb/conversation_node_store.py
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import shared.config as cfg

from langchain_core.documents import Document

from infra.repositories.mongodb.handler import MongoDBHandler
from infra.repositories.neo4j.handler import Neo4jHandler
from infra.repositories.vectordb.chunks_store import VectorDBHandler
from infra.repositories.vectordb.conversation_node_store import ConversationNodeStore
from infra.repositories.vectordb.macro_node_store import MacroNodeStore

logger = logging.getLogger("GraphNodeDB")

__all__ = ["Neo4jHandler", "VectorDBHandler", "ConversationNodeStore", "MacroNodeStore", "GraphNodeDBHandler"]


class GraphNodeDBHandler:
    @staticmethod
    def _model_slug(embedding_model: str) -> str:
        slug = embedding_model.split("/")[-1].lower()
        return slug.replace("-", "_").replace(".", "_")

    @staticmethod
    def _macro_collection_name(local_embedding_model: str) -> str:
        slug = GraphNodeDBHandler._model_slug(local_embedding_model)
        return f"macro_node_{slug}"

    @staticmethod
    def _chunks_collection_name(embedding_model: str) -> str:
        slug = GraphNodeDBHandler._model_slug(embedding_model)
        return f"microscope_chunks_{slug}"

    def __init__(self, *, chunks: Optional[List[Document]] = None) -> None:
        embedding_provider = cfg.VECTORDB_EMBEDDING_PROVIDER.lower()
        local_emb  = cfg.VECTORDB_LOCAL_EMBEDDING_MODEL
        openai_emb = cfg.VECTORDB_OPENAI_EMBEDDING_MODEL
        embedding_model = openai_emb if embedding_provider == "openai" else local_emb

        neo4j_uri      = cfg.NEO4J_URI
        neo4j_user     = cfg.NEO4J_USER
        _is_dev = os.getenv("ENV_MODE", "prod").strip().lower() == "dev"
        neo4j_password = os.getenv("NEO4J_password_dev" if _is_dev else "NEO4J_password", "")

        self.chunks = chunks
        mongo_url = os.getenv("MONGODB_URL", "")
        mongo_db  = cfg.MONGODB_DB_NAME
        self.mongodb: Optional[MongoDBHandler] = MongoDBHandler(url=mongo_url, db_name=mongo_db) if mongo_url else None
        self.graph_db: Neo4jHandler = Neo4jHandler(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        self.vector_db: VectorDBHandler = VectorDBHandler(
            persist_path="", 
            mode=cfg.CHROMA_MODE,
            host=cfg.CHROMA_SERVER_HOST,
            port=int(cfg.CHROMA_SERVER_PORT) if cfg.CHROMA_SERVER_PORT else 8000,
            embedding_model=embedding_model, embedding_provider=embedding_provider,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            chroma_tenant=cfg.CHROMA_TENANT,
            chroma_database=cfg.CHROMA_DATABASE,
            chroma_api_key=os.getenv("CHROMA_API_KEY" if cfg.CHROMA_DATABASE == "TACO_GraphNode" else "CHROMA_API_KEY_DEV", ""),
            collection_name=self._chunks_collection_name(embedding_model),
        )
        self.macro_node_store: MacroNodeStore = MacroNodeStore(
            self.vector_db.chroma_client,
            collection_name=self._macro_collection_name(local_emb),
        )

    def close(self) -> None:
        self.graph_db.close()

    def get_nodes_by_group_id(self, group_id: str, user_id: Optional[str] = None) -> List[Dict]:
        return self.graph_db.get_nodes_by_group_id(group_id, user_id)

    def store_standardized_data(self, standardized_results: List[Dict[str, Any]], source_name: str, source_id: str, user_id: str, group_id: str, chunks: Optional[List[Document]] = None) -> Dict[str, int]:
        stats = {"chunks_stored": 0, "entities_stored": 0, "edges_stored": 0, "chunk_entity_links": 0, "entities_without_chunk_id": 0}
        chunks = chunks or self.chunks or []
        chunk_index_to_id: Dict[int, str] = {}
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.metadata.get("chunk_id") or str(uuid.uuid4())
            chunk.metadata["chunk_id"] = chunk_id
            chunk_index_to_id[idx] = chunk_id

        entity_chunk_map: Dict[str, List[str]] = {}
        for batch_data in standardized_results:
            for node in batch_data.get("nodes", []):
                name = node.get("name", "")
                if not name:
                    continue
                source_chunk_id = node.get("source_chunk_id")
                if source_chunk_id is None:
                    stats["entities_without_chunk_id"] += 1
                    continue
                if isinstance(source_chunk_id, int):
                    chunk_idx = source_chunk_id
                elif isinstance(source_chunk_id, str) and source_chunk_id.isdigit():
                    chunk_idx = int(source_chunk_id)
                else:
                    chunk_idx = source_chunk_id
                if not isinstance(chunk_idx, int) or chunk_idx not in chunk_index_to_id:
                    stats["entities_without_chunk_id"] += 1
                    continue
                entity_chunk_map.setdefault(name, [])
                chunk_id = chunk_index_to_id[chunk_idx]
                if chunk_id not in entity_chunk_map[name]:
                    entity_chunk_map[name].append(chunk_id)

        payloads: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.metadata["chunk_id"]
            payloads.append({"text": chunk.page_content, "uuid": chunk_id, "source_id": source_id, "source_name": source_name, "user_id": user_id, "group_id": group_id, "chunk_index": idx, "entity_names": [name for name, cids in entity_chunk_map.items() if chunk_id in cids]})
            self.graph_db.create_chunk(uuid=chunk_id, text=chunk.page_content[:500], source_id=source_id, user_id=user_id, group_id=group_id, chunk_index=idx)

        if payloads:
            try:
                self.vector_db.add_chunks_batch(payloads)
                stats["chunks_stored"] = len(payloads)
            except Exception as exc:
                logger.warning(
                    "VectorDB chunk upsert failed (skipping, graph ingest continues): %s", exc
                )
                stats["chunks_stored"] = 0
                stats["vector_db_error"] = str(exc)

        all_entities: Dict[str, Dict[str, Any]] = {}
        all_edges: List[Dict[str, Any]] = []
        for batch_data in standardized_results:
            for node in batch_data.get("nodes", []):
                name = node.get("name", "")
                if not name:
                    continue
                entry = all_entities.setdefault(name, {"entity_uuid": str(uuid.uuid4()), "entity_name": name, "entity_types": [], "user_id": user_id, "group_id": group_id, "descriptions": [], "chunk_ids": entity_chunk_map.get(name, [])})
                node_types = node.get("type", "")
                if isinstance(node_types, str):
                    node_types = [node_types] if node_types else []
                for t in node_types:
                    if t and t not in entry["entity_types"]:
                        entry["entity_types"].append(t)
                desc = node.get("description", "")
                if desc and desc not in entry["descriptions"]:
                    entry["descriptions"].append(desc)
            for edge in batch_data.get("edges", []):
                edge_copy = edge.copy()
                edge_copy["user_id"] = user_id
                edge_copy["group_id"] = group_id
                all_edges.append(edge_copy)

        for entity_name, entity_data in all_entities.items():
            self.graph_db.merge_node(
                node={
                    "name": entity_data["entity_name"],
                    "types": entity_data["entity_types"],
                    "description": " | ".join(entity_data["descriptions"]),
                    "group_id": group_id,
                },
                source_file=source_name,
                user_id=user_id,
                chunk_id=entity_data["chunk_ids"][0] if entity_data["chunk_ids"] else None,
                source_id=source_id,
            )
            stats["entities_stored"] += 1
            for chunk_id in entity_data["chunk_ids"]:
                self.graph_db.link_entity_to_chunk(entity_name=entity_name, uuid=chunk_id, user_id=user_id, group_id=group_id)
                stats["chunk_entity_links"] += 1
        for edge in all_edges:
            self.graph_db.merge_edge(edge, user_id=user_id, source_id=source_id)
            stats["edges_stored"] += 1

        return stats

    def ingest_from_standardized(self, standardized_path: str, source_name: str = "", user_id: str = "", group_id: str = "default", chunk_id_map_path: Optional[str] = None, source_map_path: Optional[str] = None) -> Dict[str, int]:
        with open(standardized_path, "r", encoding="utf-8") as f:
            standardized_results = json.load(f)
        if chunk_id_map_path and os.path.exists(chunk_id_map_path):
            with open(chunk_id_map_path, "r", encoding="utf-8") as f:
                chunk_id_map = {str(k): v for k, v in json.load(f).items()}
        else:
            chunk_id_map = {}
        source_id = str(uuid.uuid4())
        if source_map_path and os.path.exists(source_map_path):
            with open(source_map_path, "r", encoding="utf-8") as f:
                m = json.load(f)
                if source_name in m:
                    source_id = m[source_name]

        stats = {"chunks_stored": 0, "entities_stored": 0, "edges_stored": 0, "chunk_entity_links": 0}
        all_entities: Dict[str, Dict[str, Any]] = {}
        all_edges: List[Dict[str, Any]] = []
        for batch_data in standardized_results:
            for node in batch_data.get("nodes", []):
                name = node.get("name", "")
                if not name:
                    continue
                entry = all_entities.setdefault(name, {"entity_uuid": str(uuid.uuid4()), "entity_name": name, "entity_type": node.get("type", ""), "user_id": user_id, "group_id": group_id, "descriptions": [], "chunk_ids": []})
                desc = node.get("description", "")
                if desc and desc not in entry["descriptions"]:
                    entry["descriptions"].append(desc)
                if "source_chunk_id" in node:
                    cid = chunk_id_map.get(str(node["source_chunk_id"]), str(node["source_chunk_id"]))
                    if cid not in entry["chunk_ids"]:
                        entry["chunk_ids"].append(cid)
            for edge in batch_data.get("edges", []):
                if "start" not in edge and "source" in edge:
                    edge["start"] = edge.get("source")
                edge["source"] = source_name
                edge["source_id"] = source_id
                edge["group_id"] = group_id
                edge["user_id"] = user_id
                if "source_chunk_id" in edge:
                    edge["source_chunk_id"] = chunk_id_map.get(str(edge["source_chunk_id"]), str(edge["source_chunk_id"]))
                all_edges.append(edge)

        for entity_data in all_entities.values():
            self.graph_db.merge_node(node={"name": entity_data["entity_name"], "type": entity_data["entity_type"], "description": " | ".join(entity_data["descriptions"]), "group_id": group_id}, source_file=source_name, user_id=user_id)
        for edge in all_edges:
            self.graph_db.merge_edge(edge, user_id=user_id)
            stats["edges_stored"] += 1

        return stats
