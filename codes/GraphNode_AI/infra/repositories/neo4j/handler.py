"""Neo4j graph database handler."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

logger = logging.getLogger("GraphNodeDB")


class Neo4jHandler:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def get_nodes_by_group_id(self, group_id: str, user_id: Optional[str] = None) -> List[Dict]:
        if user_id:
            q = "MATCH (n:Entity {user_id:$user_id, group_id:$group_id}) RETURN n.uuid AS uuid, n.name AS name, n.types AS types, n.descriptions AS descriptions, n.chunk_ids AS chunk_ids"
            params = {"user_id": user_id, "group_id": group_id}
        else:
            q = "MATCH (n:Entity {group_id:$group_id}) RETURN n.uuid AS uuid, n.name AS name, n.types AS types, n.descriptions AS descriptions, n.chunk_ids AS chunk_ids"
            params = {"group_id": group_id}
        with self._driver.session() as session:
            rows = session.run(q, **params)
            return [{"uuid": r.get("uuid"), "name": r.get("name"), "types": r.get("types") or [], "descriptions": r.get("descriptions") or [], "chunk_ids": r.get("chunk_ids") or []} for r in rows]

    def merge_node(self, node: Dict, source_file: str, user_id: str, chunk_id: Optional[str] = None, source_id: Optional[str] = None) -> str:
        _ = source_file
        name = node.get("name", "")
        if not name:
            return ""
        description = node.get("description", "") or ""
        types = node.get("types", node.get("type", []))
        if isinstance(types, str):
            types = [types] if types else []
        q = """
        MERGE (n:Entity {name:$name, user_id:$user_id, group_id:$group_id})
        ON CREATE SET n.uuid=$uuid, n.created_at=datetime(), n.updated_at=datetime(),
                      n.types=[], n.descriptions=[], n.source_ids=[]
        ON MATCH SET n.updated_at=datetime()
        SET n.types = coalesce(n.types, []) + [t IN $types WHERE NOT t IN coalesce(n.types, [])]
        SET n.descriptions = CASE
            WHEN size($description) > 0 AND NOT $description IN coalesce(n.descriptions, [])
            THEN coalesce(n.descriptions, []) + $description
            ELSE coalesce(n.descriptions, [])
        END
        SET n.source_ids = CASE
            WHEN $source_id IS NOT NULL AND NOT $source_id IN coalesce(n.source_ids, [])
            THEN coalesce(n.source_ids, []) + $source_id
            ELSE coalesce(n.source_ids, [])
        END
        RETURN n.uuid AS uuid
        """
        uid = str(uuid.uuid4())
        with self._driver.session() as session:
            rec = session.run(
                q,
                name=name,
                user_id=user_id,
                group_id=node.get("group_id", ""),
                uuid=uid,
                types=types,
                description=description,
                source_id=source_id,
            ).single()
            return rec["uuid"] if rec and rec.get("uuid") else uid

    def merge_edge(self, edge: Dict, user_id: str, chunk_id: Optional[str] = None, source_id: Optional[str] = None) -> str:
        _ = chunk_id
        start = edge.get("start", "")
        target = edge.get("target", "")
        etype = edge.get("type", "")
        if not start or not target or not etype:
            return ""
        q = """
        MERGE (a:Entity {name:$start, user_id:$user_id, group_id:$group_id})
        MERGE (b:Entity {name:$target, user_id:$user_id, group_id:$group_id})
        MERGE (a)-[r:REL {type:$etype, user_id:$user_id, group_id:$group_id}]->(b)
        ON CREATE SET r.uuid=$uuid, r.created_at=datetime(), r.updated_at=datetime(),
                      r.weight=$weight, r.source_ids=[]
        ON MATCH SET r.updated_at=datetime()
        SET r.source_ids = CASE
            WHEN $source_id IS NOT NULL AND NOT $source_id IN coalesce(r.source_ids, [])
            THEN coalesce(r.source_ids, []) + $source_id
            ELSE coalesce(r.source_ids, [])
        END
        RETURN r.uuid AS uuid
        """
        rid = str(uuid.uuid4())
        with self._driver.session() as session:
            rec = session.run(
                q,
                start=start,
                target=target,
                etype=etype,
                user_id=user_id,
                group_id=edge.get("group_id", ""),
                uuid=rid,
                weight=edge.get("weight", 1.0),
                source_id=source_id,
            ).single()
            return rec["uuid"] if rec and rec.get("uuid") else rid

    def create_chunk(self, uuid: str, text: str, source_id: str, user_id: str, group_id: str, chunk_index: int) -> str:
        q = "CREATE (c:Chunk {uuid:$uuid,text:$text,source_id:$source_id,user_id:$user_id,group_id:$group_id,chunk_index:$chunk_index,created_at:datetime()}) RETURN c.uuid AS uuid"
        with self._driver.session() as session:
            rec = session.run(q, uuid=uuid, text=text, source_id=source_id, user_id=user_id, group_id=group_id, chunk_index=chunk_index).single()
            return rec["uuid"] if rec and rec.get("uuid") else uuid

    def link_entity_to_chunk(self, entity_name: str, uuid: str, user_id: str, group_id: str) -> None:
        q = "MATCH (e:Entity {name:$name, user_id:$user_id, group_id:$group_id}) MATCH (c:Chunk {uuid:$uuid}) MERGE (e)-[:EXTRACTED_FROM]->(c)"
        with self._driver.session() as session:
            session.run(q, name=entity_name, user_id=user_id, group_id=group_id, uuid=uuid)

    def get_neighbors(self, entity_name: str, user_id: str, group_id: str, depth: int = 1, limit: int = 20) -> List[Dict]:
        q = f"MATCH (e:Entity {{name:$name, user_id:$user_id, group_id:$group_id}})-[:REL*1..{depth}]-(n:Entity) RETURN DISTINCT n.uuid AS uuid, n.name AS name, n.types AS types, n.descriptions AS descriptions, n.chunk_ids AS chunk_ids LIMIT $limit"
        with self._driver.session() as session:
            rows = session.run(q, name=entity_name, user_id=user_id, group_id=group_id, limit=limit)
            return [{"uuid": r.get("uuid"), "name": r.get("name"), "types": r.get("types") or [], "descriptions": r.get("descriptions") or [], "chunk_ids": r.get("chunk_ids") or [], "distance": 1} for r in rows]

    def get_one_hop_edges(self, entity_names: List[str], user_id: str, group_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        if not entity_names:
            return []
        q = "UNWIND $names AS name MATCH (e:Entity {name:name, user_id:$user_id, group_id:$group_id})-[r:REL]-(n:Entity {user_id:$user_id, group_id:$group_id}) RETURN DISTINCT e.name AS start, n.name AS end, r.type AS type LIMIT $limit"
        with self._driver.session() as session:
            rows = session.run(q, names=entity_names, user_id=user_id, group_id=group_id, limit=limit)
            return [{"start": r.get("start"), "end": r.get("end"), "types": r.get("types") or [], "descriptions": [], "evidences": [], "weight": None} for r in rows]

    def find_shortest_path(self, entity1: str, entity2: str, user_id: str, group_id: str, max_depth: int = 5) -> Optional[Dict[str, Any]]:
        q = f"MATCH (a:Entity {{name:$a, user_id:$user_id, group_id:$group_id}}), (b:Entity {{name:$b, user_id:$user_id, group_id:$group_id}}) MATCH p=shortestPath((a)-[*..{max_depth}]-(b)) RETURN [n IN nodes(p) | {{name:n.name, type:n.type}}] AS nodes, length(p) AS l"
        with self._driver.session() as session:
            rec = session.run(q, a=entity1, b=entity2, user_id=user_id, group_id=group_id).single()
            if not rec:
                return None
            return {"nodes": rec.get("nodes") or [], "relationships": [], "path_length": rec.get("l") or 0}

    def get_subgraph(self, entity_names: List[str], user_id: str, group_id: str, include_neighbors: bool = True) -> Dict[str, Any]:
        _ = include_neighbors
        return {"entities": [], "neighbors": [], "relationships": []}

    def search_entities_by_name(self, search_term: str, user_id: str, group_id: str, limit: int = 10) -> List[Dict]:
        q = "MATCH (e:Entity) WHERE e.user_id=$user_id AND e.group_id=$group_id AND toLower(e.name) CONTAINS toLower($term) RETURN e.uuid AS uuid, e.name AS name, e.type AS type LIMIT $limit"
        with self._driver.session() as session:
            rows = session.run(q, user_id=user_id, group_id=group_id, term=search_term, limit=limit)
            return [{"uuid": r.get("uuid"), "name": r.get("name"), "types": r.get("types") or [], "descriptions": []} for r in rows]

    def get_entity_by_uuid(self, entity_uuid: str) -> Optional[Dict]:
        q = "MATCH (e:Entity {uuid:$uuid}) RETURN e.uuid AS uuid, e.name AS name, e.type AS type, e.user_id AS user_id, e.group_id AS group_id"
        with self._driver.session() as session:
            rec = session.run(q, uuid=entity_uuid).single()
            if not rec:
                return None
            return {"uuid": rec.get("uuid"), "name": rec.get("name"), "type": rec.get("type"), "user_id": rec.get("user_id"), "group_id": rec.get("group_id"), "descriptions": [], "chunk_ids": []}

    def get_chunks_for_entity(self, entity_name: str, user_id: str, group_id: str) -> List[Dict]:
        q = (
            "MATCH (e:Entity {name:$name, user_id:$user_id, group_id:$group_id})-[:EXTRACTED_FROM]->(c:Chunk) "
            "OPTIONAL MATCH (e2:Entity {user_id:$user_id, group_id:$group_id})-[:EXTRACTED_FROM]->(c) "
            "RETURN c.uuid AS uuid, c.text AS text, c.source_id AS source_id, "
            "c.chunk_index AS chunk_index, collect(e2.name) AS entity_names"
        )
        with self._driver.session() as session:
            rows = session.run(q, name=entity_name, user_id=user_id, group_id=group_id)
            return [{"uuid": r.get("uuid"), "text": r.get("text"), "source_id": r.get("source_id"), "chunk_index": r.get("chunk_index"), "entity_names": list(r.get("entity_names") or [])} for r in rows]

    def get_stats(self, user_id: str, group_id: str) -> Dict[str, int]:
        q = "MATCH (e:Entity {user_id:$user_id, group_id:$group_id}) OPTIONAL MATCH (e)-[r:REL]-() OPTIONAL MATCH (c:Chunk {user_id:$user_id, group_id:$group_id}) RETURN count(DISTINCT e) AS entity_count, count(DISTINCT r) AS relationship_count, count(DISTINCT c) AS chunk_count"
        with self._driver.session() as session:
            rec = session.run(q, user_id=user_id, group_id=group_id).single()
            if not rec:
                return {"entity_count": 0, "relationship_count": 0, "chunk_count": 0}
            return {"entity_count": rec.get("entity_count", 0), "relationship_count": rec.get("relationship_count", 0), "chunk_count": rec.get("chunk_count", 0)}
