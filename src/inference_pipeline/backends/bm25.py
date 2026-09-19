"""BM25 lexical retrieval baseline, for comparison against dense backends."""

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization — good enough at this corpus
    size and avoids pulling in a full NLP tokenizer for a baseline."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Lexical search over `combined_text`, scored with Okapi BM25.

    Not a `SearchBackend` — it takes raw query text, not a query vector, so it
    doesn't share that interface with the embedding-based backends. It exists
    to be plugged into `eval_harness.evaluate_retriever()` and into RRF fusion
    (`src/eval/fusion.py`) alongside the dense backends.
    """

    def __init__(self, metadata: list[dict]):
        """Initialize BM25 over a corpus.

        Args:
            metadata: list of dicts with keys "id" and "combined_text"
        """
        self.metadata = metadata
        self.ids = [str(m.get("id")) for m in metadata]
        corpus = [tokenize(m.get("combined_text", "")) for m in metadata]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Return the top-k item ids for a query, ranked by BM25 score."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked_indices[:top_k]]

    def retrieve_with_scores(self, query: str, top_k: int = 5) -> list[dict]:
        """Same as `retrieve`, but returns full result dicts (id, score, text)
        matching the shape `SearchBackend.search()` returns, for callers that
        want to display results rather than just score them."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "id": self.ids[i],
                "score": float(scores[i]),
                "combined_text": self.metadata[i].get("combined_text"),
                "attributes": self.metadata[i].get("attributes"),
            }
            for i in ranked_indices
        ]
