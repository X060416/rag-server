"""
Sparse Retriever — BM25 keyword search
=====================================
Builds an in-memory BM25 index from all chunks in a ChromaDB collection.
Uses jieba for Chinese tokenization + simple word split for English.

The index is cached per KB and automatically invalidated when the
collection's chunk set changes (detected via id-signature).
"""

import hashlib
import logging
import re
import threading
from typing import List, Dict, Optional

import jieba
from rank_bm25 import BM25Okapi

from config import BM25_TOP_K, CHROMA_PERSIST_DIR
from vector_store import get_collection_name, get_embedding_function

from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer (jieba for Chinese, regex for English/numbers)
# ---------------------------------------------------------------------------

# Minimal stopwords — mostly function words that carry no retrieval signal
STOPWORDS = set(
    """
    的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有
    看 好 自己 这 那 与 及 或 等 并 而 但 对 从 中 里 为 之 所 被 把 让 向 于 各
    些 个 我们 你们 他们 这个 那个 这些 那些 什么 怎么 为什么 请问 一下 可以 吗 呢 吧
    a an the and or of to in on for with at by from as is are was were be been
    it its this that these those i you he she we they them what which who whom
    """.split()
)

_PUNCT_RE = re.compile(r"^[\W_]+$")


def tokenize(text: str) -> List[str]:
    """Tokenize mixed Chinese/English text.

    - Chinese: jieba word segmentation (keeps meaningful words)
    - English/numbers: kept as single tokens by jieba
    - Punctuation and stopwords are dropped
    """
    text = (text or "").lower().strip()
    if not text:
        return []

    words = jieba.lcut(text)
    tokens = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        if _PUNCT_RE.match(w):
            continue
        if w in STOPWORDS:
            continue
        tokens.append(w)
    return tokens


# ---------------------------------------------------------------------------
# BM25 index with per-KB caching
# ---------------------------------------------------------------------------

class BM25Index:
    """Thread-safe per-KB BM25 index with signature-based cache invalidation."""

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    # ---- public API ----

    def search(self, kb_id: str, query: str, top_k: int = BM25_TOP_K) -> List[Dict]:
        """BM25 keyword search. Returns [{"text","metadata","score"}] sorted desc."""
        entry = self._get_entry(kb_id)
        if entry is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = entry["index"].get_scores(tokens)
        docs = entry["docs"]
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in ranked[:top_k]:
            if score <= 0:
                continue
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(score),
            })
        return results

    def invalidate(self, kb_id: str) -> None:
        """Drop the cached index for a KB (call after delete/upload if needed)."""
        with self._lock:
            self._cache.pop(kb_id, None)

    # ---- internals ----

    def _get_entry(self, kb_id: str) -> Optional[Dict]:
        ids = self._get_ids(kb_id)
        sig = self._signature(ids)

        with self._lock:
            cached = self._cache.get(kb_id)
            if cached and cached["signature"] == sig:
                return cached

        # Rebuild (outside the lock — building is expensive)
        docs = self._get_all_docs(kb_id, ids)
        if not docs:
            return None

        corpus = [tokenize(d["text"]) for d in docs]
        entry = {
            "signature": sig,
            "index": BM25Okapi(corpus),
            "docs": docs,
        }
        with self._lock:
            self._cache[kb_id] = entry
        return entry

    @staticmethod
    def _signature(ids: List[str]) -> str:
        if not ids:
            return "empty"
        digest = hashlib.md5("".join(ids).encode("utf-8")).hexdigest()
        return f"{len(ids)}:{digest}"

    def _get_ids(self, kb_id: str) -> List[str]:
        """Cheap call: fetch only ids (no documents) to compute a signature."""
        try:
            collection_name = get_collection_name(kb_id)
            embedding_fn = get_embedding_function()
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_fn,
                persist_directory=CHROMA_PERSIST_DIR,
            )
            return vectorstore._collection.get(include=[])["ids"]
        except Exception as e:
            logger.error(f"BM25 get_ids failed for KB '{kb_id}': {e}")
            return []

    def _get_all_docs(self, kb_id: str, ids: List[str]) -> List[Dict]:
        """Fetch all chunks (text + metadata) for building the BM25 corpus."""
        try:
            collection_name = get_collection_name(kb_id)
            embedding_fn = get_embedding_function()
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_fn,
                persist_directory=CHROMA_PERSIST_DIR,
            )
            data = vectorstore._collection.get(include=["documents", "metadatas"])
            docs = []
            for i, text in enumerate(data["documents"] or []):
                meta = (data["metadatas"] or [{}])[i] or {}
                docs.append({"text": text or "", "metadata": dict(meta)})
            return docs
        except Exception as e:
            logger.error(f"BM25 get_all_docs failed for KB '{kb_id}': {e}")
            return []


# ---------------------------------------------------------------------------
# Module-level singleton (shared across requests)
# ---------------------------------------------------------------------------

_bm25 = BM25Index()


def sparse_search(kb_id: str, query: str, top_k: int = BM25_TOP_K) -> List[Dict]:
    """Module-level BM25 search entry point."""
    return _bm25.search(kb_id, query, top_k=top_k)


def invalidate(kb_id: str) -> None:
    """Invalidate the cached BM25 index for a KB."""
    _bm25.invalidate(kb_id)
