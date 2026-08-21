"""
RAG Server Configuration
All settings can be overridden via environment variables.
"""

import os

# ---- LM Studio ----
# LM Studio exposes an OpenAI-compatible API at http://localhost:1234/v1
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_LLM_MODEL = os.getenv("LM_STUDIO_LLM_MODEL", "qwen3.6-27b-mlx")
LM_STUDIO_EMBEDDING_MODEL = os.getenv("LM_STUDIO_EMBEDDING_MODEL", "text-embedding-bge-m3")

# ---- ChromaDB ----
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_PREFIX = "kb_"

# ---- Document Processing ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ---- Retrieval ----
TOP_K = int(os.getenv("TOP_K", "10"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))

# ---- Hybrid Retrieval (dense + sparse) ----
ENABLE_HYBRID_RETRIEVAL = os.getenv("ENABLE_HYBRID_RETRIEVAL", "true").lower() == "true"
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))        # BM25 candidate count
RRF_K = int(os.getenv("RRF_K", "60"))                  # RRF fusion constant (standard 60)
RRF_TOP_N = int(os.getenv("RRF_TOP_N", "15"))          # candidates kept after fusion

# ---- Cross-Encoder Reranking (ONNX, no torch needed) ----
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
# Local ONNX model: woxpas-ai/bge-reranker-v2-m3-onnx (int8, ~544MB).
# Download once with huggingface_hub snapshot_download into ./models/.
RERANKER_MODEL_DIR = os.getenv("RERANKER_MODEL_DIR", "./models/bge-reranker-v2-m3-onnx")
RERANKER_ONNX_FILE = os.getenv("RERANKER_ONNX_FILE", "onnx/model_quantized.onnx")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))     # final results after re-ranking
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "1000"))  # truncate chunk text for speed

# ---- File Upload ----
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024)))  # 50MB

# ---- External Model Connections ----
MODEL_CONNECTIONS_FILE = os.getenv(
    "MODEL_CONNECTIONS_FILE", os.path.join(CHROMA_PERSIST_DIR, "metadata", "model_connections.json")
)

# ---- Persistent File Storage ----
# Original uploaded files are kept here (per KB) to support preview & download.
# UPLOAD_DIR is only a temp scratch area during processing.
FILE_STORAGE_DIR = os.getenv("FILE_STORAGE_DIR", "./file_store")
PREVIEW_MAX_CHARS = int(os.getenv("PREVIEW_MAX_CHARS", "200000"))  # preview text cap (200K chars)

# ---- Server ----
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:3002").split(",")

# Supported file extensions for markitdown
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".csv",
    ".md", ".txt", ".html", ".htm",
    ".json", ".xml",
    ".py", ".js", ".ts", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php",
    ".swift", ".kt", ".scala", ".sh", ".sql",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff",
}
