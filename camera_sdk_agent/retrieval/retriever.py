"""
Hybrid retriever: combines vector search (Qdrant) with BM25 keyword search.

Workflow:
1. If a camera_brand filter is specified, pre-filter via Qdrant payload.
2. Vector retrieval → top-k_v (default 20).
3. BM25 over the same document set → top-k_b (default 20).
4. RRF fusion of the two ranked lists → final top-k_f (default 5).

BM25 corpus is built lazily from Qdrant on first call and refreshed
when a configurable TTL has passed.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from config import VECTOR_TOP_K, BM25_TOP_K, FINAL_TOP_K, FUSION_K
from .reranker import rrf_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Performs hybrid vector + keyword retrieval with metadata filtering."""

    def __init__(self, qdrant_store, embedder):
        """Initialize with a Qdrant store and an embedder instance.

        Args:
            qdrant_store: QdrantStore instance.
            embedder: Embedder instance.
        """
        self.store = qdrant_store
        self.embedder = embedder

        # BM25 state (lazily built)
        self._bm25_corpus: List[str] = []
        self._bm25_model: Optional[BM25Okapi] = None
        self._bm25_tokens: List[List[str]] = []
        self._bm25_ids: List[str] = []
        self._bm25_built = False

        # Cache for full chunk list (shared between vector & BM25)
        self._all_chunks_cache: Optional[List[Dict[str, Any]]] = None
        self._cache_ttl_seconds = 300  # 5 minutes

        logger.info("HybridRetriever initialized (vector_top=%d, bm25_top=%d, fusion_top=%d)",
                     VECTOR_TOP_K, BM25_TOP_K, FINAL_TOP_K)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        camera_brand: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Execute hybrid retrieval for a query.

        Args:
            query: User query string.
            camera_brand: Optional brand filter (e.g. "Basler", "HikVision").
            top_k: Override final top-k count (default from config).

        Returns:
            List of top-k results with fused scores.
        """
        if top_k is None:
            top_k = FINAL_TOP_K

        # 1. Embed the query
        query_vector = self.embedder.encode_query(query)

        # 2. Vector retrieval
        vector_results = self.store.search(
            query_vector=query_vector,
            top_k=VECTOR_TOP_K,
            camera_brand=camera_brand,
        )
        logger.debug("Vector search returned %d results", len(vector_results))

        # 3. BM25 retrieval (over the same filtered corpus)
        bm25_results = self._bm25_search(
            query=query,
            top_k=BM25_TOP_K,
            camera_brand=camera_brand,
        )
        logger.debug("BM25 search returned %d results", len(bm25_results))

        # 4. RRF fusion
        fused = rrf_fusion(
            result_sets=[vector_results, bm25_results],
            k=FUSION_K,
            final_top_k=top_k,
        )

        logger.info("Hybrid retrieval: %d vector + %d BM25 → %d fused",
                     len(vector_results), len(bm25_results), len(fused))
        return fused

    def refresh_bm25(self) -> None:
        """Force-rebuild the BM25 index from Qdrant."""
        logger.info("Building BM25 index from Qdrant collection ...")
        all_chunks = self.store.get_all_chunks_with_ids()
        if not all_chunks:
            logger.warning("No chunks found in Qdrant for BM25 indexing")
            self._bm25_corpus = []
            self._bm25_tokens = []
            self._bm25_ids = []
            self._bm25_built = True
            return

        self._bm25_corpus = [chunk["text"] for chunk in all_chunks]
        self._bm25_ids = [chunk["id"] for chunk in all_chunks]
        self._bm25_tokens = [_tokenize(text) for text in self._bm25_corpus]
        self._bm25_model = BM25Okapi(self._bm25_tokens)
        self._bm25_built = True
        self._all_chunks_cache = all_chunks
        logger.info("BM25 index built with %d documents", len(all_chunks))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        camera_brand: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run BM25 keyword search, optionally filtering by brand.

        Args:
            query: Search query.
            top_k: Number of results to return.
            camera_brand: Optional brand filter.

        Returns:
            List of result dicts with scores.
        """
        # Lazy-build BM25 if not yet ready
        if not self._bm25_built:
            self.refresh_bm25()

        if self._bm25_model is None or not self._bm25_tokens:
            return []

        query_tokens = _tokenize(query)
        scores = self._bm25_model.get_scores(query_tokens)

        # Pair (index, score) and sort descending
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for idx, score in indexed_scores:
            if camera_brand:
                chunk_meta = (
                    self._all_chunks_cache[idx]
                    if self._all_chunks_cache and idx < len(self._all_chunks_cache)
                    else None
                )
                if chunk_meta and chunk_meta.get("camera_brand", "unknown") != camera_brand:
                    continue

            chunk = {
                "id": self._bm25_ids[idx] if idx < len(self._bm25_ids) else f"bm25_{idx}",
                "score": float(score),
                "text": self._bm25_corpus[idx] if idx < len(self._bm25_corpus) else "",
            }

            # Enrich from full cache if available
            if self._all_chunks_cache and idx < len(self._all_chunks_cache):
                full = self._all_chunks_cache[idx]
                chunk.update({k: v for k, v in full.items() if k not in ("id", "text", "score")})

            results.append(chunk)

            if len(results) >= top_k:
                break

        return results


# ============================================================================
# Tokenization helper
# ============================================================================


def _tokenize(text: str) -> List[str]:
    """Mixed Chinese + English tokenizer for BM25.

    - ASCII: standard word tokenization ([a-zA-Z0-9_]+)
    - CJK: character bigrams for substring matching
    - Falls back to single CJK characters if text is short

    Args:
        text: Input text (Chinese / English / mixed).

    Returns:
        List of lowercase tokens.
    """
    import re

    tokens: List[str] = []

    # 1. Extract ASCII / English words
    ascii_tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    tokens.extend(ascii_tokens)

    # 2. Extract Chinese / CJK character runs and build bigrams
    cjk_runs: List[str] = re.findall(r"[一-鿿㐀-䶿豈-﫿]+", text)

    for run in cjk_runs:
        if len(run) == 1:
            tokens.append(run)
        else:
            # Character bigrams: "初始化相机" → ["初始","始化","化相","相机"]
            for i in range(len(run) - 1):
                tokens.append(run[i:i + 2])
            # Also add single chars for single-character queries
            tokens.extend(list(run))

    return tokens
