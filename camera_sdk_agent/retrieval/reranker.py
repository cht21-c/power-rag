"""
Reciprocal Rank Fusion (RRF) for combining multiple retrieval result sets.

RRF is a simple, effective fusion method that does not require score
calibration. Each result's rank across different retrievers contributes
to its final score:

    score(d) = Σ_{r ∈ retrievers}  1 / (k + rank_r(d))

where k controls the influence of high-ranked items (default 60).

Reference: Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet
and individual rank learning methods", SIGIR 2009.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def rrf_fusion(
    result_sets: List[List[Dict[str, Any]]],
    k: int = 60,
    final_top_k: int = 5,
    id_key: str = "id",
    text_key: str = "text",
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        result_sets: List of ranked result lists. Each inner list should
                     be sorted by relevance (best first).
        k: RRF parameter controlling high-rank emphasis (default 60).
        final_top_k: Number of top results to return after fusion.
        id_key: Dict key used to identify unique documents (default "id").
        text_key: Dict key for the document text (default "text").

    Returns:
        Merged and re-ranked list of top `final_top_k` results,
        each with an "rrf_score" field added.
    """
    # Score accumulator: {doc_id: (rrf_score, doc_dict)}
    accumulator: Dict[str, tuple] = {}

    for result_list in result_sets:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = str(doc.get(id_key, rank))
            rrf_contribution = 1.0 / (k + rank)

            if doc_id in accumulator:
                prev_score, _ = accumulator[doc_id]
                accumulator[doc_id] = (prev_score + rrf_contribution, doc)
            else:
                accumulator[doc_id] = (rrf_contribution, doc)

    # Sort by fused score descending
    sorted_items = sorted(
        accumulator.values(),
        key=lambda item: item[0],
        reverse=True,
    )

    # Build final output
    fused: List[Dict[str, Any]] = []
    for score, doc in sorted_items[:final_top_k]:
        doc_copy = dict(doc)
        doc_copy["rrf_score"] = round(score, 6)
        fused.append(doc_copy)

    logger.debug("RRF fusion: %d input sets → %d unique → top-%d",
                 len(result_sets), len(accumulator), len(fused))

    return fused
