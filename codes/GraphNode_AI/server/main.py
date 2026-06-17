"""English documentation."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GraphNode AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "microscope" / "schema"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/microscope/schemas")
def list_schemas():
    """English documentation."""
    schemas = {}
    for f in sorted(_SCHEMA_DIR.glob("ontology_schema_*.json")):
        name = f.stem.replace("ontology_schema_", "")
        schemas[name] = json.loads(f.read_text(encoding="utf-8"))
    return {"schemas": schemas}


@app.get("/microscope/schemas/{schema_name}")
def get_schema(schema_name: str):
    """English documentation."""
    path = _SCHEMA_DIR / f"ontology_schema_{schema_name}.json"
    if not path.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Schema '{schema_name}' not found")
    return json.loads(path.read_text(encoding="utf-8"))
