"""
Vector Store
Manages ChromaDB collections for knowledge bases using a custom
LM Studio-compatible embedding client.
"""

import logging
from typing import List, Dict, Optional

import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import (
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    LM_STUDIO_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_PREFIX,
    TOP_K,
)

logger = logging.getLogger(__name__)


class LMStudioEmbeddings(Embeddings): #向量化
    """
    Custom embedding client that talks directly to LM Studio's
    OpenAI-compatible /v1/embeddings endpoint.

    This avoids format incompatibilities between langchain-openai and
    some versions of LM Studio.
    """

    def __init__(
        self,
        base_url: str = LM_STUDIO_BASE_URL,
        api_key: str = LM_STUDIO_API_KEY,
        model: str = LM_STUDIO_EMBEDDING_MODEL,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    def _embed(self, texts: List[str]) -> List[List[float]]:
        # Filter and stringify inputs to satisfy LM Studio validation
        cleaned = []
        for t in texts:
            if t is None:
                t = ""
            s = str(t).strip()
            cleaned.append(s)

        if not cleaned:
            return []

        response = self.client.post(
            f"{self.base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": cleaned,
            },
        )

        if response.status_code != 200:
            logger.error(
                f"Embedding request failed: {response.status_code} - {response.text}"
            )
            response.raise_for_status()

        data = response.json().get("data", [])
        # LM Studio returns embeddings in the same order as input
        embeddings = [item["embedding"] for item in data]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]: #query向量化
        result = self._embed([text])
        return result[0] if result else []


def get_embedding_function() -> Embeddings:
    """
    Create an embedding function that connects to LM Studio's
    OpenAI-compatible embedding endpoint.
    """
    return LMStudioEmbeddings(
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY,
        model=LM_STUDIO_EMBEDDING_MODEL,
    )


def get_collection_name(kb_id: str) -> str:
    """Generate a ChromaDB collection name for a knowledge base."""
    return f"{CHROMA_COLLECTION_PREFIX}{kb_id}"


def add_documents(kb_id: str, chunks: List[dict]) -> int: #存储到chromadb
    """
    Add document chunks to a knowledge base's ChromaDB collection.
    Each chunk: {"text": str, "metadata": dict}

    Returns the number of chunks added.
    """
    collection_name = get_collection_name(kb_id)
    embedding_fn = get_embedding_function()

    documents = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in chunks
    ]

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    logger.info(f"Added {len(documents)} chunks to KB '{kb_id}' (collection: {collection_name})")
    return len(documents)


def search(
    kb_id: str,
    query: str,
    top_k: int = TOP_K,
) -> List[Dict]:
    """
    Search a knowledge base for relevant chunks.
    Returns list of {"text": str, "metadata": dict, "score": float}.
    """
    collection_name = get_collection_name(kb_id)
    embedding_fn = get_embedding_function()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    # similarity_search_with_score returns (Document, float) tuples
    # The score is a distance metric (lower = more similar for L2)
    results = vectorstore.similarity_search_with_relevance_scores(  #检索，相似度搜索
        query,
        k=top_k,
    )

    formatted = []
    for doc, score in results:
        formatted.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        })

    logger.info(f"Search '{query[:50]}...' in KB '{kb_id}': {len(formatted)} results")
    return formatted


def delete_collection(kb_id: str) -> bool:
    """Delete a knowledge base's entire ChromaDB collection."""
    collection_name = get_collection_name(kb_id)
    embedding_fn = get_embedding_function()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    vectorstore.delete_collection()
    logger.info(f"Deleted collection for KB '{kb_id}'")
    return True


def delete_documents_by_source(kb_id: str, source: str) -> bool:
    """
    Delete all chunks from a specific source file in a knowledge base.
    """
    collection_name = get_collection_name(kb_id)
    embedding_fn = get_embedding_function()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    vectorstore._collection.delete(where={"source": source})
    logger.info(f"Deleted documents from source '{source}' in KB '{kb_id}'")
    return True


def get_collection_info(kb_id: str) -> Optional[Dict]:
    """Get information about a knowledge base's collection."""
    collection_name = get_collection_name(kb_id)
    embedding_fn = get_embedding_function()

    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_fn,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        count = vectorstore._collection.count()
        return {"kb_id": kb_id, "chunk_count": count}
    except Exception as e:
        logger.error(f"Error getting collection info for KB '{kb_id}': {e}")
        return None
