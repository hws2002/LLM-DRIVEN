import os
import sys
import json
import posixpath
import re
import signal
import asyncio
import boto3
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import requests

# Ensure project root is importable when running as:
#   python server/worker.py --dev
_BOOTSTRAP_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_BOOTSTRAP_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_PROJECT_ROOT))

# English comment.
_MACRO_SRC = _BOOTSTRAP_PROJECT_ROOT / "macro" / "src"
if str(_MACRO_SRC) not in sys.path:
    sys.path.insert(0, str(_MACRO_SRC))

from shared.env_loader import load_root_env
load_root_env()

from dto.server_dto import (
    SqsEnvelope, TaskType,
    GraphGenRequestPayload, AddNodeRequestPayload, GraphGenResultPayload,
    GraphProgressPayload,
    GraphSummaryRequestPayload, GraphSummaryResultPayload,
    MicroscopeIngestRequestPayload, MicroscopeIngestResultPayload,
    MicroscopeIngestFromNodeRequest,
    MicroscopeQueryRequest, MicroscopeQueryResultPayload,
    MicroscopeSynthesizeRequest, MicroscopeSynthesizeResultPayload,
    MicroscopeRelatedQuestionsRequest, MicroscopeRelatedQuestionsResultPayload,
)

from shared import config as cfg

# English comment.
AWS_PROFILE = os.getenv("AWS_PROFILE")

# English comment.
_DEV_MODE = "--dev" in sys.argv
if _DEV_MODE:
    sys.argv.remove("--dev")
    os.environ.setdefault("CHROMA_DATABASE", "TACO_GraphNode_DEV")
    os.environ.setdefault("NEO4J_URI",  cfg.NEO4J_DEV_URI)
    os.environ.setdefault("NEO4J_USER", cfg.NEO4J_DEV_USER)
    os.environ.setdefault("NEO4J_password", os.getenv("NEO4J_password_dev", ""))
    SQS_REQUEST_QUEUE_URL = cfg.DEV_AWS_SQS_REQUEST_QUEUE_URL
    SQS_RESULT_QUEUE_URL  = cfg.DEV_AWS_SQS_RESULT_QUEUE_URL
    print("[DEV MODE] Using dev SQS queues + dev DBs")
else:
    SQS_REQUEST_QUEUE_URL = cfg.AWS_SQS_REQUEST_QUEUE_URL
    SQS_RESULT_QUEUE_URL  = cfg.AWS_SQS_RESULT_QUEUE_URL
S3_BUCKET = cfg.S3_PAYLOAD_BUCKET

# AWS client initialization
# - Local: set AWS_PROFILE and use shared credentials/config
# - Runtime (ECS/EC2/Lambda): rely on IAM role via default credential chain
if AWS_PROFILE:
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=cfg.AWS_REGION)
else:
    session = boto3.Session(region_name=cfg.AWS_REGION)

sqs = session.client("sqs", region_name=cfg.AWS_REGION)
s3 = session.client("s3", region_name=cfg.AWS_REGION)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MACRO_DIR = PROJECT_ROOT / "macro"
ADD_NODE_DIR = PROJECT_ROOT / "add_node"
INPUT_DIR = PROJECT_ROOT / "input_data"
OUTPUT_DIR = PROJECT_ROOT / "output_data"

# Ensure working directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# microscope Reset
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.api_provider import ApiProvider, normalize_provider
from microscope.call import call as microscope_call
from microscope.services import rag_service
from microscope.rag.macro_context import load_macro_summary
from infra.repositories.graph.graphnode_repository import GraphNodeDBHandler
from dto.microscope_dto import ToMicroObjectContext
# English comment.
# English comment.
_PROGRESS_STAGE_MAP: dict[str, tuple[int, str]] = {
    "embedding":  (1, "Embedding generation"),
    "keywords":   (2, "Keyword extraction"),
    "clustering": (3, "Cluster generation and mapping"),
    "edges":      (4, "Edge similarity calculation"),
    "merging":    (5, "Graph refinement"),
    "summary":    (6, "Graph summary generation"),
}

def _stage_label(step_num: int, base: str, pct: int) -> str:
    if pct == 0:   return f"[{step_num}step] {base} started"
    if pct == 100: return f"[{step_num}step] {base} completed"
    return f"[{step_num}step] {base} in progress"


# English comment.
try:
    import jieba
    jieba.initialize()
    print(f"[{datetime.now()}] jieba dictionary preloaded.")
except Exception as e:
    print(f"[{datetime.now()}] jieba preload skipped: {e}")

graphnode_db_handler = GraphNodeDBHandler()
_llm_provider = normalize_provider(cfg.MICROSCOPE_LLM_PROVIDER)
_env_key = "DEV_" + _llm_provider.upper().replace(".", "") + "_API_KEY"
rag_api_provider = ApiProvider(
    provider=_llm_provider,
    model=cfg.MICROSCOPE_LLM_MODEL,
    temperature=cfg.MICROSCOPE_LLM_TEMPERATURE,
    timeout_seconds=cfg.MICROSCOPE_LLM_TIMEOUT,
    api_key=os.getenv(_env_key, os.getenv("OPENAI_API_KEY", "")),
)
print(f"[Microscope] provider={_llm_provider}  model={cfg.MICROSCOPE_LLM_MODEL}")

def _build_api_provider(req) -> ApiProvider:
    provider = normalize_provider(req.provider or cfg.MICROSCOPE_LLM_PROVIDER)
    env_key_name = "DEV_" + provider.upper().replace(".", "") + "_API_KEY"
    api_key = req.api_key or os.getenv(env_key_name, os.getenv("OPENAI_API_KEY", ""))
    return ApiProvider(
        provider=provider,
        model=req.model or cfg.MICROSCOPE_LLM_MODEL,
        temperature=cfg.MICROSCOPE_LLM_TEMPERATURE,
        timeout_seconds=cfg.MICROSCOPE_LLM_TIMEOUT,
        api_key=api_key,
    )


def _normalize_s3_prefix(s3_key: str) -> str:
    return s3_key if s3_key.endswith("/") else f"{s3_key}/"


def _safe_relative_s3_path(prefix: str, key: str) -> Path:
    rel = key[len(prefix):] if key.startswith(prefix) else posixpath.basename(key)
    rel = rel.lstrip("/")
    parts = [part for part in rel.split("/") if part and part not in (".", "..")]
    return Path(*parts) if parts else Path(posixpath.basename(key))


def _download_s3_prefix(bucket: str, prefix: str, destination: Path, task_id: str) -> List[Path]:
    prefix = _normalize_s3_prefix(prefix)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            local_path = destination / _safe_relative_s3_path(prefix, key)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{datetime.now()}] [{task_id}]   -> {key}")
            s3.download_file(bucket, key, str(local_path))
            downloaded.append(local_path)

    if not downloaded:
        raise FileNotFoundError(f"No S3 objects found under prefix s3://{bucket}/{prefix}")
    return downloaded

async def run_subprocess_pipeline(cmd: List[str], task_id: str, cwd: Path, env: dict = None, user_id: str = None, include_summary: bool = False):
    """Run a subprocess and stream stdout/stderr in real time."""
    print(f"[{datetime.now()}] [{task_id}] Running command: {' '.join(cmd)}")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        env=env,
    )
    _active_processes.append(process)

    async def read_stream(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode().rstrip()
            # English comment.
            if prefix == task_id and decoded.startswith("STEP_START:") and user_id:
                step = decoded.split("STEP_START:", 1)[1].strip()
                entry = _PROGRESS_STAGE_MAP.get(step)
                if entry:
                    step_num, base = entry
                    await _send_progress(task_id, user_id, _stage_label(step_num, base, 0), 0)
            # English comment.
            # English comment.
            elif prefix == task_id and "PROGRESS:" in decoded and user_id:
                progress_str = decoded.split("PROGRESS:", 1)[1]
                parts = progress_str.split(":")
                # English comment.
                if len(parts) >= 2:
                    step, pct_str = parts[0], parts[1]
                    entry = _PROGRESS_STAGE_MAP.get(step)
                    try:
                        pct = int(pct_str)
                    except ValueError:
                        pct = None
                    if entry and pct is not None:
                        step_num, base = entry
                        label = _stage_label(step_num, base, pct)
                        try:
                            eta = int(parts[4]) if len(parts) >= 5 else None
                        except ValueError:
                            eta = None
                        try:
                            await _send_progress(task_id, user_id, label, pct, eta)
                        except Exception as _e:
                            print(f"[{datetime.now()}] [{task_id}] WARN: failed to send progress {pct}%: {_e}")
            print(f"[{datetime.now()}] [{prefix}] {decoded}")

    try:
        await asyncio.gather(
            read_stream(process.stdout, task_id),
            read_stream(process.stderr, f"{task_id}:ERR"),
        )
        returncode = await process.wait()
    finally:
        if process in _active_processes:
            _active_processes.remove(process)

    if returncode != 0:
        raise RuntimeError(f"Pipeline process failed with return code {returncode}")


async def _run_summary_for_graph(
    task_id: str,
    graph_s3_key: str,
    bucket: str,
    user_id: str,
    language: str = "ko",
) -> GraphSummaryResultPayload:
    """Internal helper to run graph summarization pipeline.

    Downloads graph from S3, runs src.insights.summarize subprocess, uploads result.
    Used by both handle_graph_summary() and handle_graph_generation() (when includeSummary=True).
    """
    # 1. Download graph JSON from S3
    task_input_dir = INPUT_DIR / task_id
    task_input_dir.mkdir(parents=True, exist_ok=True)
    local_graph_path = task_input_dir / "graph.json"
    print(f"[{datetime.now()}] [{task_id}] Downloading graph from s3://{bucket}/{graph_s3_key}")
    s3.download_file(bucket, graph_s3_key, str(local_graph_path))

    # 2. Run summarization pipeline
    task_output_dir = OUTPUT_DIR / task_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    local_summary_path = task_output_dir / "summary.json"

    cmd = [
        "python",
        "-m",
        "src.insights.summarize",
        "--graph",
        str(local_graph_path),
        "--output",
        str(local_summary_path),
        "--provider",
        cfg.MACRO_LLM_PROVIDER,
        "--model",
        cfg.MACRO_LLM_MODEL,
        "--verbose",
    ]

    if language:
        cmd.extend(["--language", language])

    try:
        # Run from macro/ so that src.insights.summarize resolves correctly
        # PROJECT_ROOT added to PYTHONPATH so that shared.* imports work
        summary_env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(MACRO_DIR / "src"), str(PROJECT_ROOT)])}
        await run_subprocess_pipeline(cmd, task_id, MACRO_DIR, env=summary_env, user_id=user_id)
    except Exception as e:
        print(f"[{datetime.now()}] [{task_id}] Summarization failed: {e}")
        raise e

    # 3. Upload summary result to S3
    if not local_summary_path.exists():
        raise FileNotFoundError("Summary result file not found.")

    result_s3_key = f"results/graph_summary/{user_id}/{task_id}/summary.json"
    print(f"[{datetime.now()}] [{task_id}] Uploading summary to s3://{S3_BUCKET}/{result_s3_key}")
    s3.upload_file(str(local_summary_path), S3_BUCKET, result_s3_key)

    return GraphSummaryResultPayload(
        userId=user_id,
        status="COMPLETED",
        summaryS3Key=result_s3_key,
    )


async def handle_graph_generation(task_id: str, payload: Dict[str, Any]) -> GraphGenResultPayload:
    """English documentation."""
    req = GraphGenRequestPayload(**payload)
    source_bucket = req.bucket or S3_BUCKET
    task_input_dir = INPUT_DIR / task_id
    # English comment.
    task_output_dir = OUTPUT_DIR / "macro" / task_id  # English comment.

    try:
        # 1. Download input payload from S3.
        # New: s3Key can be a prefix ending with "/" that contains input.json and raw files.
        # Legacy: s3Key is a primary input file and extraS3Keys contains additional sources.
        task_input_dir.mkdir(parents=True, exist_ok=True)

        # Prefix mode: pass the downloaded directory as --input.
        local_extra_paths: List[Path] = []
        if req.s3Key.endswith("/"):
            print(f"[{datetime.now()}] [{task_id}] Downloading prefix from s3://{source_bucket}/{req.s3Key}")
            bundle_dir = task_input_dir / "bundle"
            _download_s3_prefix(source_bucket, req.s3Key, bundle_dir, task_id)
            local_input_path = bundle_dir
        else:
            input_filename = os.path.basename(req.s3Key) or "input.json"
            local_input_path = task_input_dir / input_filename
            print(f"[{datetime.now()}] [{task_id}] Downloading input from s3://{source_bucket}/{req.s3Key}")
            s3.download_file(source_bucket, req.s3Key, str(local_input_path))

        # 2. Download legacy extra inputs from S3.
        if req.extraS3Keys:
            extra_dir = task_input_dir / "extra"
            extra_dir.mkdir(parents=True, exist_ok=True)
            for i, s3_key in enumerate(req.extraS3Keys):
                filename = os.path.basename(s3_key)
                local_path = extra_dir / filename
                if local_path.exists():
                    local_path = extra_dir / f"extra_{i}_{filename}"

                print(f"[{datetime.now()}] [{task_id}] Downloading extra input {i+1}/{len(req.extraS3Keys)} from s3://{source_bucket}/{s3_key}")
                if s3_key.endswith("/"):
                    prefix_dir = extra_dir / f"extra_prefix_{i}"
                    _download_s3_prefix(source_bucket, s3_key, prefix_dir, task_id)
                    local_extra_paths.append(prefix_dir)
                else:
                    s3.download_file(source_bucket, s3_key, str(local_path))
                    local_extra_paths.append(local_path)

        # English comment.
        if _DEV_MODE and local_input_path.is_dir() and (local_input_path / "notion.json").exists():
            task_output_dir = OUTPUT_DIR / "macro" / "notion" / task_id

        # 3. Run macro pipeline.
        task_output_dir.mkdir(parents=True, exist_ok=True)

        script_path = MACRO_DIR / "src" / "run_pipeline.py"
        config_path = MACRO_DIR / "config.yaml"

        cmd = [
            "python",
            "-u",
            str(script_path),
            "--input",
            str(local_input_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(task_output_dir),
            "--provider",
            cfg.MACRO_LLM_PROVIDER,
            "--model",
            cfg.MACRO_LLM_MODEL,
            "--skip-indexing",  # indexing handled below via MacroNodeStore (cloud ChromaDB)
            "--verbose",
            "--no-llm-edges",
        ]

        if req.numClusters:
            cmd.extend(["--num-clusters", str(req.numClusters)])
        else:
            cmd.extend(["--min-clusters", str(req.minClusters), "--max-clusters", str(req.maxClusters)])

        if req.language:
            cmd.extend(["--language", req.language])

        # Add --extra-input flags
        for extra_p in local_extra_paths:
            cmd.extend(["--extra-input", str(extra_p)])

        pipeline_env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(MACRO_DIR / "src"), str(PROJECT_ROOT)])}
        await run_subprocess_pipeline(cmd, task_id, MACRO_DIR, env=pipeline_env, user_id=req.userId, include_summary=bool(req.includeSummary))

        features_file = task_output_dir / "features.json"
        graph_final_file = task_output_dir / "graph_final.json"

        # English comment.
        result_file = task_output_dir / "graph_final.json"
        if not result_file.exists():
            raise FileNotFoundError("Result file (graph_final.json) not found after pipeline execution.")

        _is_notion = local_input_path.is_dir() and (local_input_path / "notion.json").exists()
        _s3_subpath = f"notion/{task_id}" if _is_notion else task_id
        result_s3_key = f"results/macro/{req.userId}/{_s3_subpath}/graph.json"
        print(f"[{datetime.now()}] [{task_id}] Uploading result to s3://{S3_BUCKET}/{result_s3_key}")
        s3.upload_file(str(result_file), S3_BUCKET, result_s3_key)

        # 3.1 Upload features output (optional)
        features_s3_key = None
        if features_file.exists():
            features_s3_key = f"results/macro/{req.userId}/{_s3_subpath}/features.json"
            print(f"[{datetime.now()}] [{task_id}] Uploading features to s3://{S3_BUCKET}/{features_s3_key}")
            s3.upload_file(str(features_file), S3_BUCKET, features_s3_key)

        # 3.2 Upload timing output (optional)
        timing_file = task_output_dir / "timing.json"
        if timing_file.exists():
            timing_s3_key = f"results/macro/{req.userId}/{_s3_subpath}/timing.json"
            print(f"[{datetime.now()}] [{task_id}] Uploading timing to s3://{S3_BUCKET}/{timing_s3_key}")
            s3.upload_file(str(timing_file), S3_BUCKET, timing_s3_key)

        # 4. Index embeddings into macro_node collection (cloud ChromaDB via GraphNodeDBHandler)
        indexing_error: Optional[str] = None
        macro_src = str(MACRO_DIR / "src")
        if macro_src not in sys.path:
            sys.path.insert(0, macro_src)
        from insights.storage.indexer import EmbeddingIndexer  # noqa: E402

        if features_file.exists():
            try:
                print(f"[{datetime.now()}] [{task_id}] Indexing embeddings into macro_node collection...")
                loop = asyncio.get_event_loop()
                indexer = EmbeddingIndexer(graphnode_db_handler.macro_node_store)
                await loop.run_in_executor(
                    None,
                    lambda: indexer.index_from_features(
                        features_path=features_file,
                        graph_path=graph_final_file if graph_final_file.exists() else None,
                        verbose=False,
                        user_id=req.userId,
                    ),
                )
                print(f"[{datetime.now()}] [{task_id}] macro_node indexing complete")
            except Exception as e:
                indexing_error = str(e)
                print(
                    f"[{datetime.now()}] [{task_id}] ⚠️ macro_node indexing failed "
                    f"(graph/features already uploaded): {e}"
                )

        base_result = GraphGenResultPayload(
            userId=req.userId,
            status="COMPLETED",
            resultS3Key=result_s3_key,
            featuresS3Key=features_s3_key,
            chatId=req.chatId,
        )
        if indexing_error:
            base_result.error = f"Graph succeeded, but indexing failed: {indexing_error}"

        if req.includeSummary:
            print(f"[{datetime.now()}] [{task_id}] includeSummary=True → running summary pipeline...")
            try:
                summary_result = await _run_summary_for_graph(
                    task_id=task_id,
                    graph_s3_key=result_s3_key,
                    bucket=S3_BUCKET,
                    user_id=req.userId,
                    language=req.language or "ko",
                )
                base_result.summaryS3Key = summary_result.summaryS3Key
                base_result.summaryIncluded = True
                await _send_progress(task_id, req.userId, "[6step] Graph summary generation completed", 100)
            except Exception as e:
                print(f"[{datetime.now()}] [{task_id}] ⚠️ Summary failed (graph still succeeded): {e}")
                base_result.summaryIncluded = False
                base_result.error = f"Graph succeeded, but summary failed: {e}"

        return base_result

    finally:
        shutil.rmtree(task_input_dir, ignore_errors=True)
        if not _DEV_MODE:
            shutil.rmtree(task_output_dir, ignore_errors=True)


async def handle_add_node(task_id: str, payload: Dict[str, Any]) -> GraphGenResultPayload:
    """English documentation."""
    import json as _json
    from add_node.call import run_add_node_batch_pipeline
    from add_node.utils.raw_input_adapter import (
        build_add_node_batch_from_input,
        merge_add_node_batches,
    )

    req = AddNodeRequestPayload(**payload)
    source_bucket = req.bucket or S3_BUCKET
    task_input_dir = INPUT_DIR / task_id

    try:
        # English comment.
        task_input_dir.mkdir(parents=True, exist_ok=True)
        if req.s3Key.endswith("/"):
            local_input_path = task_input_dir / "bundle"
            print(f"[{datetime.now()}] [{task_id}] Downloading add_node prefix from s3://{source_bucket}/{req.s3Key}")
            _download_s3_prefix(source_bucket, req.s3Key, local_input_path, task_id)
        else:
            input_filename = os.path.basename(req.s3Key) or "batch.json"
            local_input_path = task_input_dir / input_filename
            print(f"[{datetime.now()}] [{task_id}] Downloading add_node input from s3://{source_bucket}/{req.s3Key}")
            s3.download_file(source_bucket, req.s3Key, str(local_input_path))

        batch_data = build_add_node_batch_from_input(
            local_input_path,
            user_id=req.userId,
        )

        # Legacy/mixed mode: merge additional S3 objects or prefixes into the same batch.
        if req.extraS3Keys:
            extra_dir = task_input_dir / "extra"
            extra_dir.mkdir(parents=True, exist_ok=True)
            for i, s3_key in enumerate(req.extraS3Keys):
                filename = os.path.basename(s3_key) or f"extra_{i}"
                local_extra_path = extra_dir / filename
                print(f"[{datetime.now()}] [{task_id}] Downloading add_node extra input {i+1}/{len(req.extraS3Keys)} from s3://{source_bucket}/{s3_key}")
                if s3_key.endswith("/"):
                    local_extra_path = extra_dir / f"extra_prefix_{i}"
                    _download_s3_prefix(source_bucket, s3_key, local_extra_path, task_id)
                else:
                    s3.download_file(source_bucket, s3_key, str(local_extra_path))

                extra_batch = build_add_node_batch_from_input(
                    local_extra_path,
                    user_id=req.userId,
                )
                batch_data = merge_add_node_batches(
                    batch_data,
                    extra_batch,
                    user_id=req.userId,
                )

        normalized_batch_path = task_input_dir / "batch.json"
        normalized_batch_path.write_text(
            _json.dumps(batch_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # English comment.
        loop = asyncio.get_event_loop()
        macro_store = graphnode_db_handler.macro_node_store
        dev_out = OUTPUT_DIR / "add_node" / task_id if _DEV_MODE else None
        if dev_out:
            dev_out.mkdir(parents=True, exist_ok=True)
        result = await loop.run_in_executor(
            None,
            lambda: run_add_node_batch_pipeline(
                batch_data=batch_data,
                api_provider=rag_api_provider,
                macro_node_store=macro_store,
                dev_output_dir=dev_out,
                run_id=task_id,
                language=req.language or "ko",
            )
        )

        # English comment.
        result_s3_key = f"results/add_node/{req.userId}/{task_id}/result.json"
        result_local = task_input_dir / "result.json"
        result_local.write_text(_json.dumps(result, ensure_ascii=False), encoding="utf-8")
        print(f"[{datetime.now()}] [{task_id}] Uploading result to s3://{S3_BUCKET}/{result_s3_key}")
        s3.upload_file(str(result_local), S3_BUCKET, result_s3_key)

        if _DEV_MODE:
            shutil.copy2(str(normalized_batch_path), str(dev_out / "batch.json"))
            shutil.copy2(str(result_local), str(dev_out / "result.json"))
            print(f"[{datetime.now()}] [{task_id}] Dev output saved to {dev_out}")

        return GraphGenResultPayload(
            userId=req.userId,
            status="COMPLETED",
            resultS3Key=result_s3_key,
        )

    finally:
        shutil.rmtree(task_input_dir, ignore_errors=True)


async def handle_graph_summary(task_id: str, payload: Dict[str, Any]) -> GraphSummaryResultPayload:
    """Handle graph summary task using insights.summarize."""
    req = GraphSummaryRequestPayload(**payload)
    task_input_dir = INPUT_DIR / task_id
    task_output_dir = OUTPUT_DIR / task_id

    try:
        return await _run_summary_for_graph(
            task_id=task_id,
            graph_s3_key=req.graphS3Key,
            bucket=req.bucket or S3_BUCKET,
            user_id=req.userId,
            language=req.language or "ko",
        )
    finally:
        shutil.rmtree(task_input_dir, ignore_errors=True)
        shutil.rmtree(task_output_dir, ignore_errors=True)

async def handle_microscope_ingest(task_id: str, payload: Dict[str, Any]) -> MicroscopeIngestResultPayload:
    req = MicroscopeIngestRequestPayload(**payload)
    tmp_path = None
    try:
        suffix = Path(req.file_name).suffix
        tmp_dir = PROJECT_ROOT / "microscope" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=tmp_dir) as tmp:
            tmp_path = tmp.name
        print(f"[{datetime.now()}] [{task_id}] S3 download: s3://{req.bucket}/{req.s3_key} → {tmp_path}")
        s3.download_file(req.bucket, req.s3_key, tmp_path)
        print(f"[{datetime.now()}] [{task_id}] S3 download complete: {req.file_name}")
        to_micro = ToMicroObjectContext(
            file_path=tmp_path,
            file_name=req.file_name,
            user_id=req.user_id,
            group_id=req.group_id,
        )
        ingest_provider = _build_api_provider(req) if (req.provider or req.api_key) else rag_api_provider

        # English comment.
        block_save_dir = None
        if _DEV_MODE and req.block_mode:
            if req.output_dir:
                block_save_dir = Path(req.output_dir)
            else:
                block_save_dir = OUTPUT_DIR / "microscope" / req.user_id / task_id
            block_save_dir.mkdir(parents=True, exist_ok=True)

        _call_kwargs = dict(
            to_micro=to_micro,
            api_provider=ingest_provider,
            graph_store=graphnode_db_handler,
            schema_name=req.schema_name,
            schema=req.ontology_schema,
            user_language=req.language,
            block_mode=req.block_mode,
            block_granularity=req.block_granularity,
            source_type=req.source_type,
            generate_micro_graphs=req.generate_micro_graphs,
            save_dir=block_save_dir,
            skip_store=req.skip_store,
        )
        # English comment.
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: microscope_call(**_call_kwargs)
        )

        # English comment.
        image_map = result.pop("image_map", {})
        if req.block_mode and image_map and S3_BUCKET:
            import re as _re
            import base64 as _b64
            _ext_map = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
            s3_image_map: dict[tuple, str] = {}
            for (page_num, img_num), data in image_map.items():
                img_ext = _ext_map.get(data["mime"], "png")
                img_key = f"results/microscope/{req.user_id}/{task_id}/images/page_{page_num}_img_{img_num}.{img_ext}"
                img_bytes = _b64.b64decode(data["base64"])
                s3.put_object(Bucket=S3_BUCKET, Key=img_key, Body=img_bytes, ContentType=data["mime"])
                s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{img_key}"
                s3_image_map[(page_num, img_num)] = s3_url
            # English comment.
            def _replace_img_tag(match: "re.Match") -> str:
                page_num, img_num = int(match.group(1)), int(match.group(2))
                caption = match.group(3).strip()
                s3_url = s3_image_map.get((page_num, img_num), "")
                return f"{{{s3_url}}} <caption>{caption}</caption>"
            for block in result.get("block_graph", {}).get("blocks", []):
                if block.get("raw_text"):
                    block["raw_text"] = _re.sub(
                        r'<image page="(\d+)" index="(\d+)">\n(.*?)\n</image>',
                        _replace_img_tag, block["raw_text"], flags=_re.DOTALL,
                    )
            print(f"[{datetime.now()}] [{task_id}] {len(image_map)} images uploaded to S3")

        # English comment.
        standardized_s3_key = None
        if S3_BUCKET:
            if req.block_mode:
                standardized_s3_key = f"results/microscope/{req.user_id}/{task_id}/block_graph.json"
                standardized_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
            else:
                standardized_s3_key = f"results/microscope/{req.user_id}/{task_id}/standardized.json"
                standardized_bytes = json.dumps(result["standardized_graphs"], ensure_ascii=False).encode("utf-8")
            s3.put_object(Bucket=S3_BUCKET, Key=standardized_s3_key, Body=standardized_bytes, ContentType="application/json")
            print(f"[{datetime.now()}] [{task_id}] S3 upload: s3://{S3_BUCKET}/{standardized_s3_key}")

        if _DEV_MODE:
            if req.block_mode:
                print(f"[{datetime.now()}] [{task_id}] Block graph + intermediates saved to {block_save_dir}")
            else:
                out_dir = OUTPUT_DIR / "microscope" / req.user_id / result["source_id"]
                out_dir.mkdir(parents=True, exist_ok=True)
                _dev_files = {
                    "extracted.json":    result["extracted_graphs"],
                    "standardized.json": result["standardized_graphs"],
                    "token_usage.json":   result["token_usage"],
                    "name_mapping.json": result["name_mapping"],
                    "raw_llm.json":      result["raw_llm_outputs"],
                    "chunk_id_map.json": result["chunk_id_map"],
                }
                for fname, data in _dev_files.items():
                    (out_dir / fname).write_text(
                        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                print(f"[{datetime.now()}] [{task_id}] Dev output saved to {out_dir}")

        if req.block_mode:
            return MicroscopeIngestResultPayload(
                user_id=req.user_id, group_id=req.group_id, status="COMPLETED",
                source_id=result.get("source_id"), chunks_count=None,
                schema_name=req.schema_name, ingest_stats=None,
                standardized_s3_key=standardized_s3_key,
            )
        return MicroscopeIngestResultPayload(
            user_id=req.user_id, group_id=req.group_id, status="COMPLETED",
            source_id=result["source_id"], chunks_count=result["chunks_count"],
            schema_name=req.schema_name, ingest_stats=result["ingest_stats"],
            standardized_s3_key=standardized_s3_key,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


async def handle_microscope_ingest_from_node(task_id: str, payload: Dict[str, Any]) -> MicroscopeIngestResultPayload:
    req = MicroscopeIngestFromNodeRequest(**payload)
    if graphnode_db_handler.mongodb is None:
        raise RuntimeError("MONGODB_URL text text text.")

    if req.node_type == "note":
        doc = graphnode_db_handler.mongodb.get_note(req.node_id, req.user_id)
        if doc is None:
            raise ValueError(f"note {req.node_id} text text text text.")
        title = (doc.get("title") or f"Note {req.node_id}").strip()
        content = (doc.get("content") or "").strip()
        if not content:
            raise ValueError("note text text.")
        markdown = f"# {title}\n\n{content}\n"
    else:
        messages = graphnode_db_handler.mongodb.get_conversation_messages(req.node_id, req.user_id)
        if not messages:
            raise ValueError(f"conversation {req.node_id} text text text text text.")
        lines = [f"# Conversation {req.node_id}", ""]
        for msg in messages:
            role = (msg.get("role") or "").lower()
            if role not in ("user", "assistant"):
                continue
            prefix = "Q" if role == "user" else "A"
            content = (msg.get("content") or "").strip()
            if content:
                lines.append(f"{prefix}: {content}")
                lines.append("")
        markdown = "\n".join(lines).strip() + "\n"

    tmp_path = None
    try:
        tmp_dir = PROJECT_ROOT / "microscope" / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        import uuid as _uuid
        file_name = f"{req.node_id}.md"
        tmp_path = str(tmp_dir / f"{req.node_id}_{_uuid.uuid4().hex[:8]}.md")
        Path(tmp_path).write_text(markdown, encoding="utf-8")

        ingest_provider = _build_api_provider(req) if (req.provider or req.api_key) else rag_api_provider

        block_save_dir = None
        if _DEV_MODE:
            mode_dir = "block_mode" if req.block_mode else "non_block_mode"
            block_save_dir = OUTPUT_DIR / "microscope" / "from_graphnode" / mode_dir / task_id
            block_save_dir.mkdir(parents=True, exist_ok=True)

        _call_kwargs = dict(
            to_micro=ToMicroObjectContext(
                file_path=tmp_path,
                file_name=file_name,
                user_id=req.user_id,
                group_id=req.group_id,
            ),
            api_provider=ingest_provider,
            graph_store=graphnode_db_handler,
            schema_name=req.schema_name,
            schema=req.ontology_schema,
            user_language=req.language,
            block_mode=req.block_mode,
            block_granularity=req.block_granularity,
            generate_micro_graphs=req.generate_micro_graphs,
            save_dir=block_save_dir,
            skip_store=req.skip_store,
        )
        # English comment.
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: microscope_call(**_call_kwargs)
        )
        # English comment.
        image_map = result.pop("image_map", {})
        if req.block_mode and image_map and S3_BUCKET:
            import re as _re
            import base64 as _b64
            _ext_map = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
            s3_image_map: dict[tuple, str] = {}
            for (page_num, img_num), data in image_map.items():
                img_ext = _ext_map.get(data["mime"], "png")
                img_key = f"results/microscope/{req.user_id}/{task_id}/images/page_{page_num}_img_{img_num}.{img_ext}"
                img_bytes = _b64.b64decode(data["base64"])
                s3.put_object(Bucket=S3_BUCKET, Key=img_key, Body=img_bytes, ContentType=data["mime"])
                s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{img_key}"
                s3_image_map[(page_num, img_num)] = s3_url
            def _replace_img_tag(match: "re.Match") -> str:
                page_num, img_num = int(match.group(1)), int(match.group(2))
                caption = match.group(3).strip()
                s3_url = s3_image_map.get((page_num, img_num), "")
                return f"{{{s3_url}}} <caption>{caption}</caption>"
            for block in result.get("block_graph", {}).get("blocks", []):
                if block.get("raw_text"):
                    block["raw_text"] = _re.sub(
                        r'<image page="(\d+)" index="(\d+)">\n(.*?)\n</image>',
                        _replace_img_tag, block["raw_text"], flags=_re.DOTALL,
                    )
            print(f"[{datetime.now()}] [{task_id}] {len(image_map)} images uploaded to S3")

        # English comment.
        standardized_s3_key = None
        if S3_BUCKET:
            if req.block_mode:
                standardized_s3_key = f"results/microscope/{req.user_id}/{task_id}/block_graph.json"
                standardized_bytes = json.dumps(result, ensure_ascii=False).encode("utf-8")
                s3.put_object(Bucket=S3_BUCKET, Key=standardized_s3_key, Body=standardized_bytes, ContentType="application/json")
            else:
                standardized_s3_key = f"results/microscope/{req.user_id}/{task_id}/standardized.json"
                with tempfile.NamedTemporaryFile(delete=False, suffix=".json", dir=tmp_dir) as stmp:
                    stmp.write(json.dumps(result["standardized_graphs"], ensure_ascii=False).encode("utf-8"))
                    stmp_path = stmp.name
                try:
                    s3.upload_file(stmp_path, S3_BUCKET, standardized_s3_key)
                finally:
                    if os.path.exists(stmp_path):
                        os.remove(stmp_path)
            print(f"[{datetime.now()}] [{task_id}] S3 upload: s3://{S3_BUCKET}/{standardized_s3_key}")

        if _DEV_MODE and req.block_mode:
            print(f"[{datetime.now()}] [{task_id}] Block graph + intermediates saved to {block_save_dir}")

        return MicroscopeIngestResultPayload(
            user_id=req.user_id, group_id=req.group_id, status="COMPLETED",
            source_id=result.get("source_id", req.node_id),
            chunks_count=result.get("chunks_count", 0),
            schema_name=req.schema_name,
            ingest_stats=result.get("ingest_stats", {}),
            standardized_s3_key=standardized_s3_key,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


async def handle_microscope_query(task_id: str, payload: Dict[str, Any]) -> MicroscopeQueryResultPayload:
    req = MicroscopeQueryRequest(**payload)
    provider = _build_api_provider(req) if (req.provider or req.api_key) else rag_api_provider
    macro_summary = load_macro_summary(req.user_id, req.group_id)
    result = rag_service.run_query(
        graphnode_db_handler, provider,
        query=req.query, user_id=req.user_id, group_id=req.group_id,
        top_k=req.top_k, hop_depth=req.hop_depth, no_rag=req.no_rag,
        macro_summary=macro_summary,
    )
    return MicroscopeQueryResultPayload(
        user_id=req.user_id, group_id=req.group_id, status="COMPLETED", **result
    )


async def handle_microscope_synthesize(task_id: str, payload: Dict[str, Any]) -> MicroscopeSynthesizeResultPayload:
    req = MicroscopeSynthesizeRequest(**payload)
    provider = _build_api_provider(req) if (req.provider or req.api_key) else rag_api_provider
    result = rag_service.run_synthesize(
        graphnode_db_handler, provider,
        topic=req.topic, user_id=req.user_id, group_id=req.group_id,
        top_k=req.top_k, hop_depth=req.hop_depth, no_rag=req.no_rag,
    )
    return MicroscopeSynthesizeResultPayload(
        user_id=req.user_id, group_id=req.group_id, status="COMPLETED", **result
    )


async def handle_microscope_related_questions(task_id: str, payload: Dict[str, Any]) -> MicroscopeRelatedQuestionsResultPayload:
    req = MicroscopeRelatedQuestionsRequest(**payload)
    provider = _build_api_provider(req) if (req.provider or req.api_key) else rag_api_provider
    result = rag_service.run_related_questions(
        graphnode_db_handler, provider,
        query=req.query, user_id=req.user_id, group_id=req.group_id,
        top_k=req.top_k, hop_depth=req.hop_depth,
    )
    return MicroscopeRelatedQuestionsResultPayload(
        user_id=req.user_id, group_id=req.group_id, status="COMPLETED", **result
    )


async def send_result(task_id: str, result_payload: Any, task_type: str = TaskType.GRAPH_GENERATION_RESULT):
    """Send task result envelope to the result SQS queue."""

    # English comment.
    envelope = SqsEnvelope(
        taskType=task_type,
        payload=result_payload.model_dump(),
        taskId=task_id,
        timestamp=datetime.utcnow().isoformat(),
    )

    # English comment.
    sqs.send_message(QueueUrl=SQS_RESULT_QUEUE_URL, MessageBody=envelope.model_dump_json())
    print(f"[{datetime.now()}] [{task_id}] Result sent to SQS Result Queue ({task_type}).")


async def _send_progress(task_id: str, user_id: str, current_stage: str, progress_pct: int, eta_seconds: int = None) -> None:
    """English documentation."""
    print(f"[{datetime.now()}] [{task_id}] PROGRESS {current_stage} ({progress_pct}%)" + (f" ETA {eta_seconds}s" if eta_seconds else ""))
    if _DEV_MODE:
        return
    payload = GraphProgressPayload(
        userId=user_id,
        currentStage=current_stage,
        progressPercent=progress_pct,
        etaSeconds=eta_seconds,
    )
    await send_result(task_id, payload, task_type=TaskType.GRAPH_GENERATION_PROGRESS)


# Global state for graceful SIGTERM handling (Fargate Spot interruption)
_active_receipt_handles: List[str] = []
_active_processes: List[asyncio.subprocess.Process] = []

# ECS task scale-in protection endpoint (available in ECS runtime)
ECS_AGENT_URI = os.environ.get("ECS_AGENT_URI")


def set_task_protection(enabled: bool):
    """Enable/disable ECS task scale-in protection if ECS agent endpoint is available."""
    if not ECS_AGENT_URI:
        return

    try:
        url = f"{ECS_AGENT_URI}/task-protection/v1/state"
        res = requests.put(url, json={"ProtectionEnabled": enabled}, timeout=2)
        if res.status_code == 200:
            print(f"[{datetime.now()}] Task Protection set to: {enabled}")
        else:
            print(f"[{datetime.now()}] Failed to set task protection: {res.text}")
    except Exception as e:
        print(f"[{datetime.now()}] Error setting task protection: {e}")


async def graceful_shutdown():
    """English documentation."""
    print(f"[{datetime.now()}] SIGTERM received (Fargate Spot interruption). Shutting down gracefully...")

    # English comment.
    for proc in list(_active_processes):
        try:
            proc.terminate()
            print(f"[{datetime.now()}] Subprocess terminated.")
        except Exception as e:
            print(f"[{datetime.now()}] Failed to terminate subprocess: {e}")

    # English comment.
    for handle in list(_active_receipt_handles):
        try:
            sqs.change_message_visibility(
                QueueUrl=SQS_REQUEST_QUEUE_URL,
                ReceiptHandle=handle,
                VisibilityTimeout=0,
            )
            print(f"[{datetime.now()}] SQS message visibility reset to 0 (released for retry).")
        except Exception as e:
            print(f"[{datetime.now()}] Failed to reset SQS visibility: {e}")

    # English comment.
    set_task_protection(False)

    print(f"[{datetime.now()}] Graceful shutdown complete. Exiting.")
    os._exit(0)


async def _dispatch(task_id: str, envelope: SqsEnvelope) -> None:
    """English documentation."""

    # English comment.
    if envelope.taskType == TaskType.GRAPH_GENERATION_REQUEST:
        result = await handle_graph_generation(task_id, envelope.payload)
        await send_result(task_id, result, TaskType.GRAPH_GENERATION_RESULT)
    
    # English comment.
    elif envelope.taskType == TaskType.ADD_NODE_REQUEST:
        result = await handle_add_node(task_id, envelope.payload)
        await send_result(task_id, result, TaskType.ADD_NODE_RESULT)
    
    # English comment.
    elif envelope.taskType == TaskType.GRAPH_SUMMARY_REQUEST:
        result = await handle_graph_summary(task_id, envelope.payload)
        await send_result(task_id, result, TaskType.GRAPH_SUMMARY_RESULT)
    
    # English comment.
    elif envelope.taskType == TaskType.MICROSCOPE_INGEST_REQUEST:
        result = await handle_microscope_ingest(task_id, envelope.payload)
        await send_result(task_id, result, TaskType.MICROSCOPE_INGEST_RESULT)

    # English comment.
    elif envelope.taskType == TaskType.MICROSCOPE_INGEST_FROM_NODE_REQUEST:
        result = await handle_microscope_ingest_from_node(task_id, envelope.payload)
        await send_result(task_id, result, TaskType.MICROSCOPE_INGEST_FROM_NODE_RESULT)
    
    # English comment.
    elif envelope.taskType == TaskType.MICROSCOPE_QUERY_REQUEST:
        result = await handle_microscope_query(task_id, envelope.payload)
        await send_result(task_id, result, TaskType.MICROSCOPE_QUERY_RESULT)
    
    # English comment.
    elif envelope.taskType == TaskType.MICROSCOPE_SYNTHESIZE_REQUEST:
        result = await handle_microscope_synthesize(task_id, envelope.payload)
        await send_result(task_id, result, TaskType.MICROSCOPE_SYNTHESIZE_RESULT)
    
    # English comment.
    elif envelope.taskType == TaskType.MICROSCOPE_RELATED_QUESTIONS_REQUEST:
        result = await handle_microscope_related_questions(task_id, envelope.payload)
        await send_result(task_id, result, TaskType.MICROSCOPE_RELATED_QUESTIONS_RESULT)
    else:
        print(f"[{datetime.now()}] Unknown task type: {envelope.taskType}")


_REQUEST_TO_RESULT = {
    TaskType.GRAPH_GENERATION_REQUEST:             TaskType.GRAPH_GENERATION_RESULT,
    TaskType.ADD_NODE_REQUEST:                     TaskType.ADD_NODE_RESULT,
    TaskType.GRAPH_SUMMARY_REQUEST:                TaskType.GRAPH_SUMMARY_RESULT,
    TaskType.MICROSCOPE_INGEST_REQUEST:            TaskType.MICROSCOPE_INGEST_RESULT,
    TaskType.MICROSCOPE_INGEST_FROM_NODE_REQUEST:  TaskType.MICROSCOPE_INGEST_FROM_NODE_RESULT,
    TaskType.MICROSCOPE_QUERY_REQUEST:             TaskType.MICROSCOPE_QUERY_RESULT,
    TaskType.MICROSCOPE_SYNTHESIZE_REQUEST:        TaskType.MICROSCOPE_SYNTHESIZE_RESULT,
    TaskType.MICROSCOPE_RELATED_QUESTIONS_REQUEST: TaskType.MICROSCOPE_RELATED_QUESTIONS_RESULT,
}


async def _visibility_heartbeat(receipt_handle: str, task_id: str, interval: int = 30, extend_to: int = 90) -> None:
    """English documentation."""
    while True:
        await asyncio.sleep(interval)
        try:
            sqs.change_message_visibility(
                QueueUrl=SQS_REQUEST_QUEUE_URL,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=extend_to,
            )
        except Exception as e:
            # English comment.
            print(f"[{datetime.now()}] [{task_id}] Heartbeat stopped: {e}")
            break


async def _dispatch_and_delete(task_id: str, envelope: SqsEnvelope, receipt_handle: str, body: dict) -> None:
    """English documentation."""
    _active_receipt_handles.append(receipt_handle)
    heartbeat = asyncio.create_task(
        _visibility_heartbeat(receipt_handle, task_id)
    )
    try:
        await _dispatch(task_id, envelope)
        sqs.delete_message(QueueUrl=SQS_REQUEST_QUEUE_URL, ReceiptHandle=receipt_handle)
    except Exception as e:
        print(f"[{datetime.now()}] [{task_id}] Error processing message: {e}")
        traceback.print_exc()
        error_code = getattr(e, "error_code", None)
        task_type = body.get("taskType", "")
        payload_body = body.get("payload", {})
        err_task_type = _REQUEST_TO_RESULT.get(task_type, TaskType.GRAPH_GENERATION_RESULT)

        _MICROSCOPE_TASK_TYPES = {
            TaskType.MICROSCOPE_INGEST_REQUEST,
            TaskType.MICROSCOPE_INGEST_FROM_NODE_REQUEST,
            TaskType.MICROSCOPE_QUERY_REQUEST,
            TaskType.MICROSCOPE_SYNTHESIZE_REQUEST,
            TaskType.MICROSCOPE_RELATED_QUESTIONS_REQUEST,
        }
        if task_type in _MICROSCOPE_TASK_TYPES:
            err_res = MicroscopeIngestResultPayload(
                user_id=payload_body.get("user_id", "unknown"),
                group_id=payload_body.get("group_id", ""),
                status="FAILED",
                error=str(e),
                errorCode=error_code,
            )
        else:
            err_res = GraphGenResultPayload(
                userId=payload_body.get("userId", payload_body.get("user_id", "unknown")),
                status="FAILED",
                error=str(e),
                errorCode=error_code,
            )
        await send_result(task_id, err_res, err_task_type)
        sqs.delete_message(QueueUrl=SQS_REQUEST_QUEUE_URL, ReceiptHandle=receipt_handle)
    finally:
        heartbeat.cancel()
        if receipt_handle in _active_receipt_handles:
            _active_receipt_handles.remove(receipt_handle)


async def worker_loop():
    """English documentation."""
    concurrency = cfg.WORKER_CONCURRENCY
    semaphore = asyncio.Semaphore(concurrency)

    loop = asyncio.get_event_loop()
    try:
        loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.ensure_future(graceful_shutdown()))
    except NotImplementedError:
        pass  # English comment.

    async def _bounded_dispatch(task_id, envelope, receipt_handle, body):
        async with semaphore:
            await _dispatch_and_delete(task_id, envelope, receipt_handle, body)
        if not _active_receipt_handles:
            set_task_protection(False)

    def _sqs_poll(max_messages: int) -> list:
        """English documentation."""
        resp = sqs.receive_message(
            QueueUrl=SQS_REQUEST_QUEUE_URL,
            MaxNumberOfMessages=min(max_messages, 10),
            WaitTimeSeconds=10,
        )
        return resp.get("Messages", [])

    print(f"[{datetime.now()}] Worker started (concurrency={concurrency}). Polling Request Queue: {SQS_REQUEST_QUEUE_URL}...")
    poll_count = 0
    while True:
        # English comment.
        if len(_active_receipt_handles) >= concurrency:
            await asyncio.sleep(1)
            continue

        try:
            available = concurrency - len(_active_receipt_handles)
            # English comment.
            messages = await loop.run_in_executor(None, lambda: _sqs_poll(available))

            poll_count += 1
            if not messages:
                if poll_count % 6 == 0:  # English comment.
                    print(f"[{datetime.now()}] [idle] polling... active={len(_active_receipt_handles)}/{concurrency}")
                continue

            print(f"[{datetime.now()}] [poll] {len(messages)} message(s) received  active={len(_active_receipt_handles)}/{concurrency}")
            set_task_protection(True)

            for msg in messages:
                receipt_handle = msg.get("ReceiptHandle")
                body = json.loads(msg.get("Body", "{}"))
                try:
                    envelope = SqsEnvelope(**body)
                    task_id = envelope.taskId or msg.get("MessageId")
                    print(f"[{datetime.now()}] [{task_id}] dispatching taskType={envelope.taskType}")
                    # English comment.
                    asyncio.create_task(_bounded_dispatch(task_id, envelope, receipt_handle, body))
                except Exception as e:
                    print(f"[{datetime.now()}] Failed to parse SQS message: {e}")
                    sqs.delete_message(QueueUrl=SQS_REQUEST_QUEUE_URL, ReceiptHandle=receipt_handle)

        except Exception as e:
            print(f"[{datetime.now()}] Worker loop error: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(worker_loop())
