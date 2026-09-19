"""Reciprocal Rank Fusion — combine multiple ranked lists into one, without
needing scores to be on comparable scales (BM25 and cosine similarity aren't)."""


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse multiple ranked id lists into one.

    Each list contributes 1 / (k + rank) to an id's fused score, rank being
    1-based position in that list (ids missing from a list contribute 0 from
    it). k=60 is the standard default from the original RRF paper — it damps
    the difference between rank 1 and rank 2 so one system's top pick doesn't
    dominate purely from being marginally first.

    Args:
        rankings: one ranked list of ids per retriever
        k: RRF damping constant

    Returns:
        ids sorted by fused score, descending
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores, key=lambda item_id: scores[item_id], reverse=True)
