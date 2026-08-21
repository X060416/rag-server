"""
External model connection store.
Lets users add OpenAI-compatible endpoints (base URL + API key + model).
"""

import os
import json
import uuid
import time
from typing import List, Dict, Optional

from config import MODEL_CONNECTIONS_FILE


def _ensure_dir():
    os.makedirs(os.path.dirname(MODEL_CONNECTIONS_FILE), exist_ok=True)


def _load() -> List[Dict]:
    _ensure_dir()
    if not os.path.exists(MODEL_CONNECTIONS_FILE):
        return []
    with open(MODEL_CONNECTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(data: List[Dict]):
    _ensure_dir()
    with open(MODEL_CONNECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _mask_key(key: Optional[str]) -> str:
    """Return a masked API key safe to expose to the frontend."""
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return "*" * (len(key) - 4) + key[-4:]


def _mask(conn: Dict) -> Dict:
    """Return a copy of the connection with the API key masked."""
    c = dict(conn)
    c["api_key"] = _mask_key(c.get("api_key", ""))
    return c


def list_connections(mask_key: bool = True) -> List[Dict]:
    """List all saved external model connections."""
    conns = _load()
    return [_mask(c) for c in conns] if mask_key else conns


def get_connection(conn_id: str, include_key: bool = False) -> Optional[Dict]:
    """Get a single connection by ID."""
    for c in _load():
        if c.get("id") == conn_id:
            return c if include_key else _mask(c)
    return None


def _is_masked(api_key: str) -> bool:
    """Heuristic: if the key contains a '*', treat it as unchanged/masked."""
    return "*" in api_key


def create_connection(
    name: str,
    base_url: str,
    api_key: str = "",
    model: str = "",
    provider: str = "openai",
) -> Dict:
    """Create a new external model connection."""
    conn_id = str(uuid.uuid4())[:8]
    conn = {
        "id": conn_id,
        "name": name.strip(),
        "base_url": base_url.strip().rstrip("/"),
        "api_key": api_key or "",
        "model": model.strip(),
        "provider": (provider or "openai").strip().lower(),
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    conns = _load()
    conns.append(conn)
    _save(conns)
    return _mask(conn)


def update_connection(
    conn_id: str,
    name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[Dict]:
    """Update an existing connection. api_key is ignored if it looks masked."""
    conns = _load()
    for c in conns:
        if c.get("id") == conn_id:
            if name is not None:
                c["name"] = name.strip()
            if base_url is not None:
                c["base_url"] = base_url.strip().rstrip("/")
            if api_key is not None and api_key != "" and not _is_masked(api_key):
                c["api_key"] = api_key
            if model is not None:
                c["model"] = model.strip()
            if provider is not None:
                c["provider"] = (provider or "openai").strip().lower()
            c["updated_at"] = int(time.time())
            _save(conns)
            return _mask(c)
    return None


def delete_connection(conn_id: str) -> bool:
    """Delete a connection."""
    conns = _load()
    before = len(conns)
    conns = [c for c in conns if c.get("id") != conn_id]
    if len(conns) == before:
        return False
    _save(conns)
    return True
