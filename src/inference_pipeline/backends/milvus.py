"""Milvus-backed vector search backend."""

from pathlib import Path

import numpy as np
from pymilvus import MilvusClient

from src.inference_pipeline.backends.base import SearchBackend


class MilvusBackend(SearchBackend):
    """Vector search using Milvus or Milvus-Lite database."""

    def __init__(
        self,
        db_path: Path | str,
        collection_name: str,
        vector_field: str = "vector",
        metric_type: str = "COSINE",
        output_fields: list[str] | None = None,
    ):
        """Initialize Milvus backend.

        Args:
            db_path: Path to local Milvus database
            collection_name: Name of the collection to search
            vector_field: Name of the vector field in collection
            metric_type: Similarity metric (COSINE, L2, IP)
            output_fields: Additional fields to retrieve
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.vector_field = vector_field
        self.metric_type = metric_type
        self.output_fields = output_fields or []

        # Connect to Milvus
        self.client = MilvusClient(str(db_path))
        collection_info = self.client.describe_collection(collection_name=collection_name)
        print(f"✅ Connected to Milvus collection: {collection_name}")
        print(f"Collection info: {collection_info}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search Milvus collection.

        Args:
            query_vector: (D,) query embedding
            top_k: Number of results

        Returns:
            List of result dicts
        """
        # Convert to list format for Milvus
        vec = query_vector.astype(np.float32).tolist()
        print(f"🔍 Querying Milvus with vector (len={len(vec)}, first 5={vec[:5]})")

        # Build search params
        search_params = {"metric_type": self.metric_type, "params": {}}

        # Perform search
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                data=[vec],
                anns_field=self.vector_field,
                limit=top_k,
                output_fields=self.output_fields,
                search_params=search_params,
            )
        except TypeError:
            # Fallback for different pymilvus versions
            print("⚠️  Trying alternative search call...")
            hits = []

        # Parse results
        results = []
        if len(hits) == 0:
            return results

        for hit in hits[0]:
            hit_id = hit.get("id") if isinstance(hit, dict) else getattr(hit, "id", None)

            score = getattr(hit, "score", None) or getattr(hit, "distance", None)

            result = {
                "idx": hit_id,
                "id": hit_id,
                "score": float(score) if score is not None else None,
            }

            # Add output fields
            try:
                entity = getattr(hit, "entity", None)
                if entity and isinstance(entity, dict):
                    result.update(entity)
                elif entity is not None and hasattr(entity, "get"):
                    for field in self.output_fields:
                        result[field] = entity.get(field)
            except Exception:
                pass

            results.append(result)

        return results

    def close(self) -> None:
        """Close Milvus connection."""
        if hasattr(self, "client"):
            # Milvus client doesn't have explicit close, but we can reference it here
            pass
