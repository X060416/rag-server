"""
Knowledge Base Metadata Store
Simple JSON-based storage for KB and document metadata.
"""

import os
import json
import uuid
import time
from typing import List, Dict, Optional
from config import CHROMA_PERSIST_DIR

# Metadata file path
KB_META_DIR = os.path.join(CHROMA_PERSIST_DIR, "metadata")
KB_META_FILE = os.path.join(KB_META_DIR, "knowledge_bases.json")
DOC_META_FILE = os.path.join(KB_META_DIR, "documents.json")


def _ensure_dirs():
    os.makedirs(KB_META_DIR, exist_ok=True)


def _load_json(filepath: str) -> List[Dict]:
    _ensure_dirs()
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(filepath: str, data: list):
    _ensure_dirs()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---- Knowledge Base CRUD ----

def create_kb(name: str, description: str = "") -> Dict:
    """Create a new knowledge base record."""
    kb_id = str(uuid.uuid4())[:8]
    kb = {
        "id": kb_id,
        "name": name,
        "description": description,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "document_count": 0,
    }
    kbs = _load_json(KB_META_FILE)
    kbs.append(kb)
    _save_json(KB_META_FILE, kbs)
    return kb


def list_kbs() -> List[Dict]:
    """List all knowledge bases."""
    return _load_json(KB_META_FILE)


def get_kb(kb_id: str) -> Optional[Dict]:
    """Get a single knowledge base by ID."""
    kbs = _load_json(KB_META_FILE)
    for kb in kbs:
        if kb["id"] == kb_id:
            return kb
    return None


def delete_kb(kb_id: str) -> bool:
    """Delete a knowledge base record."""
    kbs = _load_json(KB_META_FILE)
    kbs = [kb for kb in kbs if kb["id"] != kb_id]
    _save_json(KB_META_FILE, kbs)

    # Also delete associated documents
    docs = _load_json(DOC_META_FILE)
    docs = [doc for doc in docs if doc["kb_id"] != kb_id]
    _save_json(DOC_META_FILE, docs)
    return True


def update_kb(kb_id: str, name: str = None, description: str = None) -> Optional[Dict]:
    """Update a knowledge base's metadata."""
    kbs = _load_json(KB_META_FILE)
    for kb in kbs:
        if kb["id"] == kb_id:
            if name:
                kb["name"] = name
            if description is not None:
                kb["description"] = description
            kb["updated_at"] = int(time.time())
            _save_json(KB_META_FILE, kbs)
            return kb
    return None


# ---- Document CRUD ----

def add_document(kb_id: str, filename: str, file_size: int, chunk_count: int, doc_id: Optional[str] = None) -> Dict:
    """Add a document record after processing."""
    if doc_id is None:
        doc_id = str(uuid.uuid4())[:8]
    doc = {
        "id": doc_id,
        "kb_id": kb_id,
        "filename": filename,
        "file_size": file_size,
        "chunk_count": chunk_count,
        "created_at": int(time.time()),
    }
    docs = _load_json(DOC_META_FILE)
    docs.append(doc)
    _save_json(DOC_META_FILE, docs)

    # Update KB document count
    kbs = _load_json(KB_META_FILE)
    for kb in kbs:
        if kb["id"] == kb_id:
            kb["document_count"] = kb.get("document_count", 0) + 1
            kb["updated_at"] = int(time.time())
    _save_json(KB_META_FILE, kbs)

    return doc


def list_documents(kb_id: str) -> List[Dict]:
    """List all documents in a knowledge base."""
    docs = _load_json(DOC_META_FILE)
    return [doc for doc in docs if doc["kb_id"] == kb_id]


def get_document(doc_id: str) -> Optional[Dict]:
    """Get a single document record by ID."""
    docs = _load_json(DOC_META_FILE)
    for doc in docs:
        if doc["id"] == doc_id:
            return doc
    return None


def delete_document(doc_id: str) -> Optional[Dict]:
    """Delete a document record. Returns the deleted doc (for source-based vector deletion)."""
    docs = _load_json(DOC_META_FILE)
    deleted = None
    remaining = []
    for doc in docs:
        if doc["id"] == doc_id:
            deleted = doc
        else:
            remaining.append(doc)
    _save_json(DOC_META_FILE, remaining)

    if deleted:
        # Update KB document count
        kbs = _load_json(KB_META_FILE)
        for kb in kbs:
            if kb["id"] == deleted["kb_id"]:
                kb["document_count"] = max(0, kb.get("document_count", 0) - 1)
                kb["updated_at"] = int(time.time())
        _save_json(KB_META_FILE, kbs)

    return deleted
