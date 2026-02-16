"""Local in-memory search backend using numpy."""
import numpy as np

from src.inference_pipeline.backends.base import SearchBackend


class LocalBackend(SearchBackend):
    """In-memory search using normalized embeddings and cosine similarity."""

    def __init__(self, embeddings: np.ndarray, metadata: list[dict]):
        """Initialize local backend.

        Args:
            embeddings: (N, D) array of embeddings
            metadata: List of N metadata dicts with keys: id, combined_text, attributes
        """
        self.embeddings = embeddings.astype(np.float32)
        self.metadata = metadata

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._embeddings_normalized = self.embeddings / norms

        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"Metadata count ({len(metadata)}) != embeddings rows ({embeddings.shape[0]})"
            )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search using cosine similarity via dot product with normalized arrays.

        Args:
            query_vector: (D,) query embedding (should be normalized)
            top_k: Number of results

        Returns:
            List of result dicts
        """
        # Compute similarity scores
        similarities = np.dot(self._embeddings_normalized, query_vector)

        # Get top-k indices
        indices = np.argsort(-similarities)[:top_k]

        # Build results
        results = []
        for idx in indices:
            meta = self.metadata[int(idx)] if idx < len(self.metadata) else {}
            results.append(
                {
                    "idx": int(idx),
                    "id": meta.get("id"),
                    "score": float(similarities[idx]),
                    "combined_text": meta.get("combined_text"),
                    "attributes": meta.get("attributes"),
                }
            )
        return results

    def close(self) -> None:
        """No cleanup needed for local backend."""
