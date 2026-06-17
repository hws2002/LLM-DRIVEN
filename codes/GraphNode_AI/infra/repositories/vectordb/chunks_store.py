"""ChromaDB vector store for document chunks."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction, SentenceTransformerEmbeddingFunction


class VectorDBHandler:
    DEFAULT_CHUNKS_COLLECTION = "microscope_chunks"

    def __init__(self, persist_path: Optional[str] = None, mode: str = "local", host: Optional[str] = None, port: Optional[int] = None, embedding_model: Optional[str] = None, embedding_function: Optional[Any] = None, embedding_provider: str = "local", openai_api_key: Optional[str] = None, reset_collections: bool = False, chroma_tenant: Optional[str] = None, chroma_database: Optional[str] = None, chroma_api_key: Optional[str] = None, collection_name: Optional[str] = None) -> None:
        if embedding_function is None and embedding_model:
            embedding_function = OpenAIEmbeddingFunction(api_key=openai_api_key, model_name=embedding_model) if embedding_provider == "openai" else SentenceTransformerEmbeddingFunction(model_name=embedding_model)
        if mode == "cloud":
            self._client = chromadb.CloudClient(
                tenant=chroma_tenant,
                database=chroma_database,
                api_key=chroma_api_key,
            )
        elif mode == "server":
            self._client = chromadb.HttpClient(host=host, port=port)
        else:  # "local"
            self._client = chromadb.PersistentClient(path=persist_path)
        chunks_collection = collection_name or self.DEFAULT_CHUNKS_COLLECTION
        if reset_collections:
            try:
                self._client.delete_collection(name=chunks_collection)
            except Exception:
                pass
        self._chunks_collection = self._client.get_or_create_collection(name=chunks_collection, embedding_function=embedding_function, metadata={"hnsw:space": "cosine"})

    @property
    def chroma_client(self):
        return self._client

    @staticmethod
    def _normalize_str_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if str(v)]
        if isinstance(value, str):
            return [v for v in value.split(",") if v]
        return [str(value)]

    @staticmethod
    def _serialize_str_list(value: Any) -> str:
        return ",".join(VectorDBHandler._normalize_str_list(value))

    def add_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[str]:
        if not chunks:
            return []
        documents = []
        metadatas = []
        ids = []
        for chunk in chunks:
            ids.append(chunk["uuid"])
            documents.append(chunk["text"])
            metadatas.append({"source_id": chunk.get("source_id", ""), "source_name": chunk.get("source_name", ""), "user_id": chunk["user_id"], "group_id": chunk["group_id"], "chunk_index": chunk.get("chunk_index", 0), "entity_names": self._serialize_str_list(chunk.get("entity_names", [])), "created_at": int(time.time())})
        self._chunks_collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        return ids

    def retrieve_chunks(self, query: str, user_id: str, group_id: str, n_results: int = 5, source_name: Optional[str] = None) -> List[Dict[str, Any]]:
        clauses = []
        if user_id:
            clauses.append({"user_id": {"$eq": user_id}})
        if group_id:
            clauses.append({"group_id": {"$eq": group_id}})
        if source_name:
            clauses.append({"source_name": {"$eq": source_name}})
        where = clauses[0] if len(clauses) == 1 else {"$and": clauses} if len(clauses) >= 2 else None
        kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": n_results}
        if where is not None:
            kwargs["where"] = where
        results = self._chunks_collection.query(**kwargs)
        out: List[Dict[str, Any]] = []
        if results and results.get("documents") and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                out.append({"text": doc, "uuid": results["ids"][0][i], "source_id": results["metadatas"][0][i].get("source_id"), "source_name": results["metadatas"][0][i].get("source_name"), "chunk_index": results["metadatas"][0][i].get("chunk_index"), "entity_names": self._normalize_str_list(results["metadatas"][0][i].get("entity_names", [])), "distance": results["distances"][0][i] if results.get("distances") else None})
        return out
