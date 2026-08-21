"""
Retriever — Advanced RAG retrieval chain
========================================
Pipeline:
  1. Dense retrieval  (ChromaDB vector similarity via bge-m3 embeddings)
  2. Sparse retrieval (BM25 keyword search via jieba + rank_bm25)
  3. RRF fusion       (Reciprocal Rank Fusion merges both rankings)
  4. Cross-encoder reranking (bge-reranker-v2-m3 re-scores candidates)
  5. Threshold filter & formatting

Every step can be toggled off via config so the system degrades gracefully
back to plain dense retrieval if a dependency is missing.
"""

import logging
from typing import List, Dict

from config import (
    TOP_K,
    SCORE_THRESHOLD,
    ENABLE_HYBRID_RETRIEVAL,
    ENABLE_RERANKING,
    BM25_TOP_K,
    RRF_K,
    RRF_TOP_N,
)
from vector_store import search as dense_search
from sparse_retriever import sparse_search
import reranker

logger = logging.getLogger(__name__)

# The system prompt template that instructs the LLM to use retrieved context 注入上下文到system prompt 
RAG_SYSTEM_PROMPT = """You are a helpful assistant with access to a knowledge base.
Use the following retrieved context to answer the user's question.
If the context doesn't contain relevant information, say so honestly.
Always cite the source when using information from the context.

--- Retrieved Context ---
{context}
--- End of Context ---
"""


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def rrf_fusion(
    dense: List[Dict],
    sparse: List[Dict],
    k: int = RRF_K,
    top_n: int = RRF_TOP_N,
) -> List[Dict]:
    """
    Reciprocal Rank Fusion: merge two ranked lists by reciprocal ranks.
    Score(doc) = sum(1 / (k + rank_i)) over every list the doc appears in.
    Documents found by BOTH retrievers get a boost — this is the key benefit
    of hybrid retrieval.
    """
    scores: Dict[str, float] = {}
    items: Dict[str, Dict] = {}

    for rank, r in enumerate(dense):
        key = r["text"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        items[key] = r

    for rank, r in enumerate(sparse):
        key = r["text"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        items[key] = r

    ordered = sorted(items.items(), key=lambda kv: scores[kv[0]], reverse=True)
    fused = []
    for key, _ in ordered[:top_n]:
        r = dict(items[key])  # shallow copy so we don't mutate shared dicts
        r["score"] = scores[key]  # unified RRF score (0..1-ish)
        r["score_source"] = "rrf"
        fused.append(r)

    logger.info(f"RRF fusion: {len(dense)} dense + {len(sparse)} sparse -> {len(fused)} candidates")
    return fused


# ---------------------------------------------------------------------------
# Main retrieval chain
# ---------------------------------------------------------------------------

def retrieve_context(kb_id: str, query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    Advanced RAG retrieval:
      dense → (sparse → RRF) → cross-encoder rerank → threshold filter
    """
    # 1. Dense retrieval — fetch extra candidates so reranking has room to work
    dense_candidates = dense_search(kb_id, query, top_k=max(BM25_TOP_K, top_k * 3))

    if ENABLE_HYBRID_RETRIEVAL and dense_candidates:
        # 2. Sparse retrieval
        sparse_candidates = sparse_search(kb_id, query, top_k=BM25_TOP_K)
        # 3. RRF fusion
        candidates = rrf_fusion(dense_candidates, sparse_candidates)
    else:
        candidates = dense_candidates

    if ENABLE_RERANKING and len(candidates) > 1:
        # 4. Cross-encoder re-ranking (scores become 0..1 relevance)
        try:
            candidates = reranker.rerank(query, candidates, top_k=top_k)
        except Exception as e:
            logger.error(f"Reranking failed, falling back to RRF order: {e}")
            candidates = candidates[:top_k]
    else:
        candidates = candidates[:top_k]

    # 5. Threshold filter — only meaningful when scores are comparable
    #    (dense relevance or cross-encoder 0..1). RRF fusion scores have a
    #    different scale (~0.03), so skip the threshold in that case.
    score_sources = {r.get("score_source", "dense") for r in candidates}
    use_threshold = not (score_sources == {"rrf"})

    if use_threshold:
        filtered = [r for r in candidates if r["score"] >= SCORE_THRESHOLD]
        if len(filtered) == 0 and len(candidates) > 0:
            # If nothing passes threshold, keep top results anyway
            logger.warning(
                f"No results passed threshold {SCORE_THRESHOLD}, "
                f"returning top {min(3, len(candidates))} unfiltered"
            )
            filtered = candidates[:3]
    else:
        filtered = candidates

    logger.info(
        f"Retrieved {len(filtered)} relevant chunks for query '{query[:50]}...' "
        f"(hybrid={ENABLE_HYBRID_RETRIEVAL}, rerank={ENABLE_RERANKING})"
    )
    return filtered


def format_context(results: List[Dict]) -> str: #把检索到的chunks格式化成带来源信息的文本
    """
    Format retrieved chunks into a context string for the LLM prompt.
    Chunk text already includes 'Source:' and 'Section:' prefixes from document_processor,
    so we only need to prepend an index and relevance score here.
    """
    if not results:
        return "No relevant context found."

    formatted_parts = []
    for i, result in enumerate(results, 1):
        score = result["score"]
        # Source/Section 已经写进 chunk 文本里，这里只加序号+分数
        header = f"[{i}] Relevance: {score:.2f}"
        formatted_parts.append(f"{header}\n{result['text']}")

    return "\n\n---\n\n".join(formatted_parts)


def build_rag_prompt(kb_id: str, query: str, top_k: int = TOP_K) -> Dict:
    """
    Full RAG retrieval: query the knowledge base, format context,
    and return both the formatted context and the system prompt.

    Returns: {
        "context": formatted context string,
        "system_prompt": RAG system prompt with context injected,
        "sources": list of source names used,
        "chunks": raw chunk results,
    }
    """
    results = retrieve_context(kb_id, query, top_k)
    context = format_context(results)
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)

    sources = list(set(r["metadata"].get("source", "unknown") for r in results))

    return {
        "context": context,
        "system_prompt": system_prompt,
        "sources": sources,
        "chunks": results,
    }
