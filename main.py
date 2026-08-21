"""
RAG Server - FastAPI Application
Provides document upload, knowledge base management, and RAG retrieval APIs.
"""

import os
import uuid
import json
import logging
import shutil
import mimetypes
import urllib.request
import ssl
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import (
    HOST, PORT, CORS_ORIGINS, UPLOAD_DIR, FILE_STORAGE_DIR, PREVIEW_MAX_CHARS, MAX_FILE_SIZE,
    LM_STUDIO_BASE_URL, LM_STUDIO_LLM_MODEL, LM_STUDIO_EMBEDDING_MODEL,
)
from document_processor import process_document, is_supported, convert_to_markdown
from vector_store import add_documents, delete_collection, delete_documents_by_source, get_collection_info
from retriever import build_rag_prompt
import kb_store
import model_connections

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---- App ----
app = FastAPI(title="ChatUI RAG Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FILE_STORAGE_DIR, exist_ok=True)


# Text-like extensions: read raw content directly for preview
TEXT_EXTENSIONS = {
    ".md", ".txt", ".html", ".htm", ".json", ".xml", ".csv",
    ".py", ".js", ".ts", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php",
    ".swift", ".kt", ".scala", ".sh", ".sql",
}
# Image extensions: preview via <img> pointing at the download endpoint
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".gif"}


def _stored_file_path(doc: dict) -> Optional[str]:
    """Locate the persisted original file for a document record."""
    kb_dir = os.path.join(FILE_STORAGE_DIR, doc["kb_id"])
    if not os.path.isdir(kb_dir):
        return None
    expected = os.path.join(kb_dir, f"{doc['id']}_{os.path.basename(doc['filename'])}")
    if os.path.exists(expected):
        return expected
    # Fallback: scan for any file with the doc id prefix
    prefix = doc["id"] + "_"
    for name in os.listdir(kb_dir):
        if name.startswith(prefix):
            p = os.path.join(kb_dir, name)
            if os.path.isfile(p):
                return p
    return None


# ---- Models ----

class CreateKBRequest(BaseModel):
    name: str
    description: str = ""


class UpdateKBRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class QueryRequest(BaseModel):
    kb_id: str
    query: str
    top_k: Optional[int] = None


class CreateConnectionRequest(BaseModel):
    name: str
    base_url: str
    api_key: str = ""
    model: str = ""
    provider: str = "openai"


class UpdateConnectionRequest(BaseModel):
    name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    lm_studio_url: str
    llm_model: str
    embedding_model: str


# ---- Health ----

@app.get("/api/rag/health")
async def health():
    return HealthResponse(
        status="ok",
        lm_studio_url=LM_STUDIO_BASE_URL,
        llm_model=LM_STUDIO_LLM_MODEL,
        embedding_model=LM_STUDIO_EMBEDDING_MODEL,
    )


# ---- External Model Connections ----

@app.get("/api/rag/connections")
async def list_connections():
    """List saved external model connections (API keys masked)."""
    return {"connections": model_connections.list_connections(mask_key=True)}


@app.post("/api/rag/connections")
async def create_connection(req: CreateConnectionRequest):
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Connection name is required")
    if not req.base_url.strip():
        raise HTTPException(status_code=400, detail="Base URL is required")
    conn = model_connections.create_connection(
        name=req.name,
        base_url=req.base_url,
        api_key=req.api_key,
        model=req.model,
        provider=req.provider,
    )
    logger.info(f"Created model connection: {conn['id']} - {conn['name']}")
    return {"status": "success", "connection": conn}


@app.get("/api/rag/connections/{conn_id}")
async def get_connection(conn_id: str, include_key: bool = False):
    conn = model_connections.get_connection(conn_id, include_key=include_key)
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    return conn


@app.put("/api/rag/connections/{conn_id}")
async def update_connection(conn_id: str, req: UpdateConnectionRequest):
    conn = model_connections.update_connection(
        conn_id=conn_id,
        name=req.name,
        base_url=req.base_url,
        api_key=req.api_key,
        model=req.model,
        provider=req.provider,
    )
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")
    logger.info(f"Updated model connection: {conn_id}")
    return {"status": "success", "connection": conn}


@app.delete("/api/rag/connections/{conn_id}")
async def delete_connection(conn_id: str):
    if not model_connections.delete_connection(conn_id):
        raise HTTPException(status_code=404, detail="Connection not found")
    logger.info(f"Deleted model connection: {conn_id}")
    return {"status": "deleted", "connection_id": conn_id}


@app.post("/api/rag/connections/{conn_id}/test")
def test_connection(conn_id: str):
    """Test a connection by calling its /models endpoint."""
    conn = model_connections.get_connection(conn_id, include_key=True)
    if not conn:
        raise HTTPException(status_code=404, detail="Connection not found")

    base = conn["base_url"].rstrip("/")
    url = f"{base}/models"
    headers = {}
    if conn.get("api_key"):
        if conn.get("provider") == "azure-openai":
            headers["api-key"] = conn["api_key"]
        else:
            headers["Authorization"] = f"Bearer {conn['api_key']}"

    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("id", m.get("model", str(m))) for m in data.get("data", [])]
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- Knowledge Base CRUD ----

@app.get("/api/rag/kb")
async def list_knowledge_bases():
    kbs = kb_store.list_kbs()
    # Enrich with chunk counts from ChromaDB
    for kb in kbs:
        info = get_collection_info(kb["id"])
        kb["chunk_count"] = info["chunk_count"] if info else 0
    return kbs


@app.post("/api/rag/kb")
async def create_knowledge_base(req: CreateKBRequest):
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    kb = kb_store.create_kb(name=req.name.strip(), description=req.description)
    logger.info(f"Created KB: {kb['id']} - {kb['name']}")
    return kb


@app.patch("/api/rag/kb/{kb_id}")
async def update_knowledge_base(kb_id: str, req: UpdateKBRequest):
    kb = kb_store.update_kb(kb_id, name=req.name, description=req.description)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


@app.delete("/api/rag/kb/{kb_id}")
async def delete_knowledge_base(kb_id: str):
    # Delete vectors first
    try:
        delete_collection(kb_id)
    except Exception as e:
        logger.warning(f"Failed to delete collection for KB {kb_id}: {e}")
    # Delete metadata
    kb_store.delete_kb(kb_id)
    # Delete stored original files
    kb_dir = os.path.join(FILE_STORAGE_DIR, kb_id)
    if os.path.isdir(kb_dir):
        shutil.rmtree(kb_dir, ignore_errors=True)
        logger.info(f"Removed file store for KB {kb_id}")
    logger.info(f"Deleted KB: {kb_id}")
    return {"status": "deleted", "kb_id": kb_id}


# ---- Document Upload ----

@app.post("/api/rag/upload")
async def upload_document(
    file: UploadFile = File(...),
    kb_id: str = Form(...),
):
    if not kb_id:
        raise HTTPException(status_code=400, detail="kb_id is required")

    # Check file extension
    filename = file.filename or "unknown"
    if not is_supported(filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {filename}. Supported: pdf, docx, pptx, xlsx, md, txt, html, code files, etc."
        )

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB")

    # Save to temp file
    file_path = os.path.join(UPLOAD_DIR, f"{kb_id}_{filename}")
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        # Process: markitdown -> chunk -> embed -> store
        chunks = process_document(file_path, source_name=filename)

        if not chunks:
            raise HTTPException(status_code=422, detail="No content extracted from document")

        chunk_count = add_documents(kb_id, chunks)

        # Save metadata (doc_id generated here so the stored file can reuse it)
        doc_id = str(uuid.uuid4())[:8]
        doc = kb_store.add_document(
            kb_id=kb_id,
            filename=filename,
            file_size=len(content),
            chunk_count=chunk_count,
            doc_id=doc_id,
        )

        # Persist the original file for preview/download
        safe_name = os.path.basename(filename)
        kb_dir = os.path.join(FILE_STORAGE_DIR, kb_id)
        os.makedirs(kb_dir, exist_ok=True)
        stored_path = os.path.join(kb_dir, f"{doc_id}_{safe_name}")
        shutil.move(file_path, stored_path)

        logger.info(f"Uploaded '{filename}' to KB '{kb_id}': {chunk_count} chunks")
        return {
            "status": "success",
            "document": doc,
            "chunk_count": chunk_count,
            "stored_path": stored_path,
        }

    except Exception as e:
        logger.error(f"Error processing file '{filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Clean up temp file (already moved to file_store on success)
        if os.path.exists(file_path):
            os.remove(file_path)


# ---- Document Management ----

@app.get("/api/rag/documents")
async def list_documents(kb_id: str = Query(...)):
    docs = kb_store.list_documents(kb_id)
    return docs


@app.delete("/api/rag/documents/{doc_id}")
async def delete_document(doc_id: str):
    deleted = kb_store.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete stored original file
    try:
        path = _stored_file_path(deleted)
        if path and os.path.exists(path):
            os.remove(path)
            logger.info(f"Removed stored file for doc {doc_id}")
    except Exception as e:
        logger.warning(f"Failed to remove stored file for doc {doc_id}: {e}")

    # Delete vectors by source
    try:
        delete_documents_by_source(deleted["kb_id"], deleted["filename"])
    except Exception as e:
        logger.warning(f"Failed to delete vectors for '{deleted['filename']}': {e}")

    return {"status": "deleted", "document": deleted}


@app.get("/api/rag/documents/{doc_id}/download")
async def download_document(doc_id: str):
    """Download the original uploaded file."""
    doc = kb_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    path = _stored_file_path(doc)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Original file not found on disk")
    media_type = mimetypes.guess_type(doc["filename"])[0] or "application/octet-stream"
    return FileResponse(path, filename=doc["filename"], media_type=media_type)


@app.get("/api/rag/documents/{doc_id}/preview")
async def preview_document(doc_id: str):
    """Preview a document's content (text extraction or image)."""
    doc = kb_store.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    path = _stored_file_path(doc)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Original file not found on disk")

    base = {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "file_size": doc["file_size"],
    }
    ext = os.path.splitext(doc["filename"])[1].lower()

    # Image: point the frontend at the download endpoint as the image URL
    if ext in IMAGE_EXTENSIONS:
        return {
            **base,
            "type": "image",
            "image_url": f"/api/rag/documents/{doc_id}/download",
        }

    # Text-like: read raw content directly
    if ext in TEXT_EXTENSIONS:
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                text = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read file: {str(e)}")
    else:
        # Office / PDF / etc: extract via markitdown
        try:
            text = convert_to_markdown(path)
        except Exception as e:
            logger.error(f"Preview extraction failed for '{doc['filename']}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Preview extraction failed: {str(e)}")

    truncated = False
    if len(text) > PREVIEW_MAX_CHARS:
        text = text[:PREVIEW_MAX_CHARS] + "\n\n[... preview truncated ...]"
        truncated = True

    return {
        **base,
        "type": "text",
        "text": text,
        "char_count": len(text),
        "truncated": truncated,
    }


# ---- RAG Query ----

@app.post("/api/rag/query")
async def rag_query(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    # Check KB exists
    kb = kb_store.get_kb(req.kb_id)
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    result = build_rag_prompt(
        kb_id=req.kb_id,
        query=req.query,
        top_k=req.top_k or 5,
    )

    return {
        "kb_id": req.kb_id,
        "query": req.query,
        "context": result["context"],
        "system_prompt": result["system_prompt"],
        "sources": result["sources"],
        "chunk_count": len(result["chunks"]),
        "chunks": [
            {
                "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                "source": c["metadata"].get("source", "unknown"),
                "section": c["metadata"].get("section", ""),
                "score": round(c["score"], 4),
            }
            for c in result["chunks"]
        ],
    }


# ---- Extract Text (for chat-level file upload, no KB storage) ----

@app.post("/api/rag/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from a document using markitdown.
    Does NOT store in any knowledge base — just returns the text.
    Used for chat-level file attachments.
    """
    filename = file.filename or "unknown"

    # Check file extension
    if not is_supported(filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {filename}. Supported: pdf, docx, pptx, xlsx, md, txt, html, code files, etc."
        )

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE // (1024*1024)}MB")

    # Save to temp file
    file_path = os.path.join(UPLOAD_DIR, f"extract_{filename}")
    with open(file_path, "wb") as f:
        f.write(content)

    try:
    # 图片走 OCR，其他走 markitdown
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}:
            from document_processor import ocr_image
            extracted_text = ocr_image(file_path)
        else:
            from document_processor import convert_to_markdown
            extracted_text = convert_to_markdown(file_path)

        # Truncate if too long (avoid exceeding LLM context window)
        max_chars = 12000
        truncated = False
        if len(extracted_text) > max_chars:
            extracted_text = extracted_text[:max_chars] + "\n\n[... document truncated ...]"
            truncated = True

        logger.info(f"Extracted {len(extracted_text)} chars from '{filename}' (chat attachment)")
        return {
            "status": "success",
            "filename": filename,
            "text": extracted_text,
            "char_count": len(extracted_text),
            "truncated": truncated,
        }



    except Exception as e:
        logger.error(f"Error extracting text from '{filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


# ---- Entry Point ----

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting RAG server on {HOST}:{PORT}")
    logger.info(f"LM Studio URL: {LM_STUDIO_BASE_URL}")
    logger.info(f"Embedding model: {LM_STUDIO_EMBEDDING_MODEL}")
    logger.info(f"LLM model: {LM_STUDIO_LLM_MODEL}")
    uvicorn.run(app, host=HOST, port=PORT)
