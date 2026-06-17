from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TaskType:
    # macro
    GRAPH_GENERATION_REQUEST = "GRAPH_GENERATION_REQUEST"
    GRAPH_GENERATION_RESULT = "GRAPH_GENERATION_RESULT"
    GRAPH_GENERATION_PROGRESS = "GRAPH_GENERATION_PROGRESS"
    ADD_NODE_REQUEST = "ADD_NODE_REQUEST"
    ADD_NODE_RESULT = "ADD_NODE_RESULT"
    ADD_CONVERSATION_REQUEST = "ADD_CONVERSATION_REQUEST"
    GRAPH_SUMMARY_REQUEST = "GRAPH_SUMMARY_REQUEST"
    GRAPH_SUMMARY_RESULT = "GRAPH_SUMMARY_RESULT"
    # microscope
    MICROSCOPE_INGEST_REQUEST = "MICROSCOPE_INGEST_REQUEST"
    MICROSCOPE_INGEST_RESULT = "MICROSCOPE_INGEST_RESULT"
    MICROSCOPE_INGEST_FROM_NODE_REQUEST = "MICROSCOPE_INGEST_FROM_NODE_REQUEST"
    MICROSCOPE_INGEST_FROM_NODE_RESULT = "MICROSCOPE_INGEST_FROM_NODE_RESULT"
    MICROSCOPE_QUERY_REQUEST = "MICROSCOPE_QUERY_REQUEST"
    MICROSCOPE_QUERY_RESULT = "MICROSCOPE_QUERY_RESULT"
    MICROSCOPE_SYNTHESIZE_REQUEST = "MICROSCOPE_SYNTHESIZE_REQUEST"
    MICROSCOPE_SYNTHESIZE_RESULT = "MICROSCOPE_SYNTHESIZE_RESULT"
    MICROSCOPE_RELATED_QUESTIONS_REQUEST = "MICROSCOPE_RELATED_QUESTIONS_REQUEST"
    MICROSCOPE_RELATED_QUESTIONS_RESULT = "MICROSCOPE_RELATED_QUESTIONS_RESULT"


class SqsEnvelope(BaseModel):
    taskType: str
    payload: Dict[str, Any]
    timestamp: Optional[str] = None
    taskId: Optional[str] = None


# ── macro payload DTOs ───────────────────────────────────────────────────────

class GraphProgressPayload(BaseModel):
    userId: str
    currentStage: str
    progressPercent: int
    etaSeconds: Optional[int] = None  # English comment.


class GraphGenRequestPayload(BaseModel):
    chatId: Optional[str] = None
    s3Key: str
    bucket: Optional[str] = None
    userId: str
    numClusters: Optional[int] = None
    minClusters: Optional[int] = 3
    maxClusters: Optional[int] = 8
    includeSummary: Optional[bool] = True
    language: Optional[str] = "zh"
    # Multi-source support
    inputType: str = "auto"         # "auto" | "chatgpt" | "markdown" | "multi" | "pdf" | "docx" | "pptx"
    extraS3Keys: List[str] = Field(default_factory=list)  # Legacy: extra source objects merged with s3Key


class AddNodeRequestPayload(BaseModel):
    userId: str
    s3Key: str
    bucket: Optional[str] = None
    beBaseUrl: Optional[str] = None
    internalServiceToken: Optional[str] = None
    inputType: str = "auto"  # "auto" | "batch" | "raw" | "pdf" | "docx" | "pptx" | "markdown"
    extraS3Keys: List[str] = Field(default_factory=list)
    language: Optional[str] = "zh"


class AddNodeBatchRequest(BaseModel):
    """English documentation."""
    userId: str
    existingClusters: List[Dict[str, Any]] = []
    conversations: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []


class GraphGenResultPayload(BaseModel):
    userId: str
    status: Literal["COMPLETED", "FAILED"]
    resultS3Key: Optional[str] = None
    featuresS3Key: Optional[str] = None
    error: Optional[str] = None
    errorCode: Optional[str] = None
    chatId: Optional[str] = None
    summaryS3Key: Optional[str] = None
    summaryIncluded: bool = False


class GraphSummaryRequestPayload(BaseModel):
    userId: str
    chatId: Optional[str] = None
    graphS3Key: str
    bucket: Optional[str] = None
    vectorDbS3Key: Optional[str] = None
    language: Optional[str] = None


class GraphSummaryResultPayload(BaseModel):
    userId: str
    status: Literal["COMPLETED", "FAILED"]
    summaryS3Key: Optional[str] = None
    chatId: Optional[str] = None
    error: Optional[str] = None
    errorCode: Optional[str] = None


# English comment.

class MicroscopeQueryRequest(BaseModel):
    query: str
    user_id: str = ""
    group_id: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    top_k: int = 5
    hop_depth: int = 1
    no_rag: bool = False


class MicroscopeSynthesizeRequest(BaseModel):
    topic: str
    user_id: str = ""
    group_id: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    top_k: int = 5
    hop_depth: int = 1
    no_rag: bool = False


class MicroscopeRelatedQuestionsRequest(BaseModel):
    query: str
    user_id: str = ""
    group_id: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    top_k: int = 5
    hop_depth: int = 1


# ── microscope HTTP response DTOs (server/main.py) ───────────────────────────

class MicroscopeIngestResult(BaseModel):
    source_name: str
    source_id: str
    chunks_count: int
    schema_name: Optional[str]
    ingest_stats: Dict[str, Any]


class MicroscopeIngestFromNodeRequest(BaseModel):
    model_config = {"populate_by_name": True}

    user_id: str
    node_id: str
    node_type: Literal["note", "conversation"]
    group_id: str = ""
    schema_name: Optional[str] = None
    ontology_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    language: Optional[str] = "zh"
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    block_mode: bool = False
    block_granularity: str = "medium"
    generate_micro_graphs: bool = Field(default=False, alias="generateMicroGraphs")
    skip_store: bool = False


class MicroscopeQueryResult(BaseModel):
    answer: str
    context: str
    chunks: List[Dict[str, Any]]


class MicroscopeSynthesizeResult(BaseModel):
    answer: str
    context: str
    chunks: List[Dict[str, Any]]


class MicroscopeRelatedQuestionsResult(BaseModel):
    questions: str
    entities: List[str]


# ── microscope SQS result payload DTOs (server/worker.py) ────────────────────

class MicroscopeIngestRequestPayload(BaseModel):
    model_config = {"populate_by_name": True}

    user_id: str
    group_id: str
    s3_key: str
    bucket: str
    file_name: str
    schema_name: Optional[str] = None
    ontology_schema: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    language: Optional[str] = "zh"
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    block_mode: bool = False
    block_granularity: str = "medium"   # "coarse" | "medium" | "fine"
    source_type: str = "chat"           # "chat" | "note"
    generate_micro_graphs: bool = Field(default=False, alias="generateMicroGraphs")
    skip_store: bool = False
    output_dir: Optional[str] = None    # English comment.


class MicroscopeIngestResultPayload(BaseModel):
    user_id: str
    group_id: str
    status: Literal["COMPLETED", "FAILED"]
    source_id: Optional[str] = None
    chunks_count: Optional[int] = None
    schema_name: Optional[str] = None
    ingest_stats: Optional[Dict[str, Any]] = None
    standardized_s3_key: Optional[str] = None
    error: Optional[str] = None
    errorCode: Optional[str] = None


class MicroscopeQueryResultPayload(BaseModel):
    user_id: str
    group_id: str
    status: Literal["COMPLETED", "FAILED"]
    answer: Optional[str] = None
    context: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    errorCode: Optional[str] = None


class MicroscopeSynthesizeResultPayload(BaseModel):
    user_id: str
    group_id: str
    status: Literal["COMPLETED", "FAILED"]
    answer: Optional[str] = None
    context: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    errorCode: Optional[str] = None


class MicroscopeRelatedQuestionsResultPayload(BaseModel):
    user_id: str
    group_id: str
    status: Literal["COMPLETED", "FAILED"]
    questions: Optional[str] = None
    entities: Optional[List[str]] = None
    error: Optional[str] = None
    errorCode: Optional[str] = None
