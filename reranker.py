"""
Cross-Encoder Reranker (ONNX Runtime)
=====================================
Uses the ONNX export of BAAI/bge-reranker-v2-m3 (int8 quantized) to re-rank
candidate chunks by (query, chunk) relevance.

Why ONNX instead of sentence-transformers?
  This machine is macOS Intel (x86_64) + Python 3.13 — PyTorch has no wheel
  for that combination (macOS Intel wheels stopped at torch 2.2.x). The ONNX
  export runs on onnxruntime (already installed via RapidOCR), no torch needed.

Model: woxpas-ai/bge-reranker-v2-m3-onnx (int8 quantized, ~544MB)
  - First use downloads it to ./models/bge-reranker-v2-m3-onnx
  - Lazy-loaded and cached after that

Inference:
  tokenizer(query, doc) -> input_ids/attention_mask -> logit -> sigmoid -> 0..1
"""

import logging
import os
import threading
from typing import List, Dict, Optional

import numpy as np
import onnxruntime as ort

from config import (
    RERANKER_MODEL_DIR,
    RERANKER_ONNX_FILE,
    RERANK_TOP_K,
    RERANK_MAX_CHARS,
)

logger = logging.getLogger(__name__)

_state: Optional[Dict] = None
_state_lock = threading.Lock()


def _load() -> Dict:
    """Lazy-load tokenizer + ONNX session (thread-safe, cached)."""
    global _state
    if _state is not None:
        return _state

    with _state_lock:
        if _state is None:
            from transformers import AutoTokenizer

            onnx_path = os.path.join(RERANKER_MODEL_DIR, RERANKER_ONNX_FILE)
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(
                    f"Reranker ONNX model not found at {onnx_path}. "
                    f"Download it first (see README)."
                )

            logger.info(f"Loading ONNX reranker tokenizer from {RERANKER_MODEL_DIR} ...")
            tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_DIR)

            logger.info(f"Loading ONNX reranker session from {onnx_path} ...")
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            input_names = {i.name for i in session.get_inputs()}

            _state = {
                "session": session,
                "tokenizer": tokenizer,
                "input_names": input_names,
            }
            logger.info("Cross-encoder reranker (ONNX) loaded.")
    return _state


def is_available() -> bool:
    """Whether the model files exist and onnxruntime can be imported."""
    try:
        import onnxruntime  # noqa: F401
        return os.path.exists(os.path.join(RERANKER_MODEL_DIR, RERANKER_ONNX_FILE))
    except ImportError:
        return False


def rerank(
    query: str,
    candidates: List[Dict],
    top_k: int = RERANK_TOP_K,
) -> List[Dict]:
    """
    Re-rank candidate chunks with the cross-encoder.

    Args:
        query: the user's question
        candidates: list of {"text", "metadata", "score"} (e.g. RRF-fused results)
        top_k: how many results to keep after re-ranking

    Returns:
        Re-ranked candidates (top_k), each with:
          - "score": cross-encoder relevance (sigmoid, 0..1)
          - "score_source": "cross_encoder"
    """
    if not candidates:
        return []

    st = _load()
    session, tokenizer, input_names = st["session"], st["tokenizer"], st["input_names"]

    # Truncate chunk text to keep inference fast & within model context
    queries = [query] * len(candidates)
    docs = [c["text"][:RERANK_MAX_CHARS] for c in candidates]

    try:
        enc = tokenizer(
            queries,
            docs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        feed = {k: v for k, v in enc.items() if k in input_names}
        logits = session.run(None, feed)[0].reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid -> 0..1
    except Exception as e:
        # Fallback: keep original order if inference fails
        logger.error(f"Cross-encoder predict failed: {e}")
        return candidates[:top_k]

    for c, p in zip(candidates, probs):
        c["score"] = float(p)
        c["score_source"] = "cross_encoder"

    ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]
