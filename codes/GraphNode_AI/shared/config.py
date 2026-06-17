"""English documentation."""

import os

# ── ENV MODE ──────────────────────────────────────────────────────────────────
# English comment.
_IS_DEV = os.getenv("ENV_MODE", "prod").strip().lower() == "dev"

# ── AWS ───────────────────────────────────────────────────────────────────────
AWS_REGION        = os.getenv("AWS_REGION",        "ap-northeast-2")
S3_PAYLOAD_BUCKET = os.getenv("S3_PAYLOAD_BUCKET", "taco5-graphnode-graphdata-s3")

AWS_SQS_REQUEST_QUEUE_URL     = os.getenv("AWS_SQS_REQUEST_QUEUE_URL",     "https://sqs.ap-northeast-2.amazonaws.com/571721033550/taco-graphnode-request-graph-sqs")
AWS_SQS_RESULT_QUEUE_URL      = os.getenv("AWS_SQS_RESULT_QUEUE_URL",      "https://sqs.ap-northeast-2.amazonaws.com/571721033550/taco-graphnode-response-graph-sqs")
DEV_AWS_SQS_REQUEST_QUEUE_URL = os.getenv("DEV_AWS_SQS_REQUEST_QUEUE_URL", "https://sqs.ap-northeast-2.amazonaws.com/571721033550/taco-graphnode-request-graph-sqs-dev")
DEV_AWS_SQS_RESULT_QUEUE_URL  = os.getenv("DEV_AWS_SQS_RESULT_QUEUE_URL",  "https://sqs.ap-northeast-2.amazonaws.com/571721033550/taco-graphnode-response-graph-sqs-dev")

# English comment.
CHROMA_TENANT   = os.getenv("CHROMA_TENANT",   "f9334ca3-a2bc-429e-a318-de2eb114d68d")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "TACO_GraphNode_DEV" if _IS_DEV else "TACO_GraphNode")
CHROMA_MODE     = os.getenv("CHROMA_MODE",     "cloud").lower()
if CHROMA_MODE == "local" and os.getenv("CHROMA_SERVER_HOST"):
    CHROMA_MODE = "server"  # Map to internal HttpClient mode
CHROMA_SERVER_HOST = os.getenv("CHROMA_SERVER_HOST", None)
CHROMA_SERVER_PORT = os.getenv("CHROMA_SERVER_PORT", "8000")

VECTORDB_EMBEDDING_PROVIDER     = os.getenv("VECTORDB_EMBEDDING_PROVIDER",     "local")   # local | openai
VECTORDB_LOCAL_EMBEDDING_MODEL  = os.getenv("VECTORDB_LOCAL_EMBEDDING_MODEL",  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
VECTORDB_OPENAI_EMBEDDING_MODEL = os.getenv("VECTORDB_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── LLM API base URLs ─────────────────────────────────────────────────────────
OPENAI_BASE_URL      = os.getenv("OPENAI_BASE_URL",      "https://api.openai.com/v1")
ZAI_BASE_URL         = os.getenv("ZAI_BASE_URL",         "https://open.bigmodel.cn/api/paas/v4")
GROQ_BASE_URL        = os.getenv("GROQ_BASE_URL",        "https://api.groq.com/openai/v1")
OPENROUTER_BASE_URL  = os.getenv("OPENROUTER_BASE_URL",  "https://openrouter.ai/api/v1")

# ── Neo4j ─────────────────────────────────────────────────────────────────────
NEO4J_URI  = os.getenv("NEO4J_URI",  "neo4j+s://1b4a6670.databases.neo4j.io" if _IS_DEV else "neo4j+s://149cc0e2.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "1b4a6670" if _IS_DEV else "149cc0e2")

NEO4J_DEV_URI  = os.getenv("NEO4J_DEV_URI",  "neo4j+s://1b4a6670.databases.neo4j.io")
NEO4J_DEV_USER = os.getenv("NEO4J_DEV_USER", "1b4a6670")

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "test")

# ── LLM (Microscope) ──────────────────────────────────────────────────────────
MICROSCOPE_LLM_PROVIDER = os.getenv("MICROSCOPE_LLM_PROVIDER", "openai")
MICROSCOPE_LLM_MODEL    = os.getenv("MICROSCOPE_LLM_MODEL",    "gpt-5-mini")
MICROSCOPE_LLM_TEMPERATURE = float(os.getenv("MICROSCOPE_LLM_TEMPERATURE", "0.0"))
MICROSCOPE_LLM_TIMEOUT     = float(os.getenv("MICROSCOPE_LLM_TIMEOUT",     "300.0"))
# ── worker ────────────────────────────────────────────────────────────────────
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "5"))

# English comment.
MACRO_LLM_PROVIDER = os.getenv("MACRO_LLM_PROVIDER", "openai")
MACRO_LLM_MODEL    = os.getenv("MACRO_LLM_MODEL",    "gpt-5-mini")
# English comment.
MICROSCOPE_CHUNK_SIZE       = int(os.getenv("MICROSCOPE_CHUNK_SIZE",       "400"))
MICROSCOPE_CHUNK_OVERLAP    = int(os.getenv("MICROSCOPE_CHUNK_OVERLAP",    "80"))
# English comment.
MICROSCOPE_BATCH_MAX_TOKENS = int(os.getenv("MICROSCOPE_BATCH_MAX_TOKENS", "10000"))

# English comment.
ADDNODE_EMBEDDING_MODEL           = os.getenv("ADDNODE_EMBEDDING_MODEL",           "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ADDNODE_KEYWORD_METHOD            = os.getenv("ADDNODE_KEYWORD_METHOD",            "keybert")   # keybert | ngram | langchain
ADDNODE_KEYWORD_TOP_N             = int(os.getenv("ADDNODE_KEYWORD_TOP_N",         "10"))
ADDNODE_NGRAM_MAX                 = int(os.getenv("ADDNODE_NGRAM_MAX",             "3"))
ADDNODE_QA_CLUSTERING_MODE        = os.getenv("ADDNODE_QA_CLUSTERING_MODE",        "all_qa")   # all_qa | hdbscan
ADDNODE_MIN_CLUSTER_SIZE          = int(os.getenv("ADDNODE_MIN_CLUSTER_SIZE",      "2"))
ADDNODE_MERGE_DISTANCE_THRESHOLD  = float(os.getenv("ADDNODE_MERGE_DISTANCE_THRESHOLD",  "0.15"))
ADDNODE_EDGE_SIMILARITY_THRESHOLD = float(os.getenv("ADDNODE_EDGE_SIMILARITY_THRESHOLD", "0.6"))
ADDNODE_EDGE_TOP_K                = int(os.getenv("ADDNODE_EDGE_TOP_K",            "5"))
ADDNODE_EDGE_FETCH_TOP_K          = int(os.getenv("ADDNODE_EDGE_FETCH_TOP_K",      "20"))
ADDNODE_EDGE_FALLBACK_ENABLED     = os.getenv("ADDNODE_EDGE_FALLBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
ADDNODE_EDGE_FALLBACK_TOP_K       = int(os.getenv("ADDNODE_EDGE_FALLBACK_TOP_K",   "20"))
ADDNODE_NEW_CLUSTER_GUARD_ENABLED = os.getenv("ADDNODE_NEW_CLUSTER_GUARD_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
ADDNODE_NEW_CLUSTER_GUARD_THRESHOLD = float(os.getenv("ADDNODE_NEW_CLUSTER_GUARD_THRESHOLD", "0.30"))
ADDNODE_NEW_CLUSTER_GUARD_EMBED_WEIGHT = float(os.getenv("ADDNODE_NEW_CLUSTER_GUARD_EMBED_WEIGHT", "0.6"))
ADDNODE_NEW_CLUSTER_GUARD_KEYWORD_WEIGHT = float(os.getenv("ADDNODE_NEW_CLUSTER_GUARD_KEYWORD_WEIGHT", "0.4"))

# ── shared text core profile ─────────────────────────────────────────────────
# English comment.
# English comment.
TEXT_CORE_PROFILE_MACRO   = os.getenv("TEXT_CORE_PROFILE_MACRO", "balanced").strip().lower()
TEXT_CORE_PROFILE_ADDNODE = os.getenv("TEXT_CORE_PROFILE_ADDNODE", "balanced").strip().lower()
