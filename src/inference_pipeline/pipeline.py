import json
import sys
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config.inference_config import InferenceConfig

try:
    # pymilvus is optional; if present we can use a Milvus-backed search
    from pymilvus import Collection, connections
except Exception:
    connections = None
    Collection = None


def find_embeddings_file(embeddings_dir: Path, embedding_type: str = "sentence") -> Path | None:
    """Find an embeddings file in embeddings_dir.

    embedding_type: "sentence" -> text embeddings (endswith text.npy)
                    "clip" -> clip embeddings (endswith clip.npy)
    """
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    if embedding_type == "sentence":
        pat = "sentence*"
    elif embedding_type == "clip":
        pat = "clip*"
    else:
        raise ValueError(f"Unknown embedding_type {embedding_type}")
    files = list(embeddings_dir.glob(pat))
    return files[0] if files else None


def load_embeddings(embeddings_dir: Path, embedding_type: str = "sentence") -> np.ndarray:
    path = find_embeddings_file(embeddings_dir, embedding_type)
    if path is None:
        raise FileNotFoundError(
            f"No embeddings file found for type={embedding_type} in {embeddings_dir}"
        )
    arr = np.load(str(path))
    # ensure 2D
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    print(f"Loaded embeddings from {path}, shape={arr.shape}")
    return arr


def load_metadata(dataset_path: Path) -> list[dict]:
    items = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


class InferencePipeline:
    """Lightweight inference pipeline: load embeddings + dataset, compute query embedding and return top-k nearest items.

    Contract:
    - inputs: query text (str)
    - outputs: list[dict] with keys (id, score, combined_text, attributes)
    - error modes: raises FileNotFoundError if embeddings/metadata not present; raises RuntimeError if embedding model not available
    """

    def __init__(self, config: InferenceConfig):
        """Initialize pipeline from config.

        Args:
            config: InferenceConfig instance with all settings
        """
        self.config = config

        # Model configuration
        self.model_name = config.model.model_name
        self.device = config.device.device
        self.embedding_type = config.model.embedding_type

        # Backend configuration
        self.backend = config.vector_db.backend

        # Paths from config
        self.embeddings_dir = Path(config.paths.embeddings_dir)
        self.dataset_path = Path(config.dataset_path)

        # Load numpy embeddings and metadata for local backend
        self.embeddings = None
        self.metadata = None
        if self.backend == "local":
            self.embeddings = load_embeddings(self.embeddings_dir, self.embedding_type)
            self.metadata = load_metadata(self.dataset_path)
            if len(self.metadata) != self.embeddings.shape[0]:
                print(
                    f"Warning: metadata count {len(self.metadata)} != embeddings rows {self.embeddings.shape[0]}"
                )

            # normalize embeddings for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._embeddings_normalized = self.embeddings / norms

        # Milvus params from config
        self.milvus_collection_name = config.vector_db.collection_name
        self.milvus_vector_field = config.vector_db.vector_field
        self.milvus_output_fields = config.vector_db.output_fields or []
        self.milvus_host = config.vector_db.host
        self.milvus_port = config.vector_db.port
        self.milvus_db_path = config.vector_db.local_path

        # Model lazy-loaded
        self._model = None

        # Lazy-initialize milvus collection handle if requested
        self._milvus_collection = None
        if self.backend == "milvus_lite":
            self.milvus_client = MilvusClient(str(self.milvus_db_path))
            collection_info = self.milvus_client.describe_collection(
                collection_name=self.milvus_collection_name
            )
            print(f"✅ Connected to Milvus collection: {self.milvus_collection_name}")
            print(f"Collection info: {collection_info}")
        elif self.backend == "milvus":
            raise NotImplementedError("Milvus backend not yet implemented")

    def _ensure_model(self):
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError(
                    "sentence-transformers is not installed; install with `pip install sentence-transformers`"
                )
            self._model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> np.ndarray:
        """Return a single normalized embedding for the input text."""
        self._ensure_model()
        emb = self._model.encode([text], convert_to_numpy=True)[0].astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        """Query the dataset for nearest textual matches to `text`.

        Returns list of dicts {id, score, combined_text, attributes, idx}
        """
        q = self.embed_text(text)
        if self.backend == "local":
            # cosine similarity via dot product with normalized arrays
            sims = np.dot(self._embeddings_normalized, q)
            # get top k indices
            idxs = np.argsort(-sims)[:top_k]
            results = []
            for idx in idxs:
                meta = self.metadata[idx] if idx < len(self.metadata) else {}
                results.append(
                    {
                        "idx": int(idx),
                        "id": meta.get("id"),
                        "score": float(sims[idx]),
                        "combined_text": meta.get("combined_text"),
                        "attributes": meta.get("attributes"),
                    }
                )
            return results
        elif self.backend == "milvus_lite":
            return self._query_milvus(q, top_k=top_k)
        elif self.backend == "milvus":
            raise NotImplementedError("Milvus backend not yet implemented")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _query_milvus(self, q: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search Milvus collection and return results translated to our result format."""
        # ensure query vector is python list
        vec = q.astype(np.float32).tolist()
        print(f"🔍 Querying Milvus with vector (len={len(vec)}, first 5 vals={vec[:5]})")

        # build search params — for cosine we use COSINE on normalized vectors
        search_params = {"metric_type": self.config.vector_db.metric_type.upper(), "params": {}}

        # perform search
        try:
            hits = self.milvus_client.search(
                collection_name=self.milvus_collection_name,
                data=[vec],  # List containing your vector
                anns_field=self.milvus_vector_field,
                limit=top_k,
                output_fields=self.milvus_output_fields,
                search_params=search_params,
            )
        except TypeError:
            # some pymilvus versions name the arg `param` differently; try without named args
            print("⚠️  TypeError encountered, trying alternative search call...")
            hits = self.milvus_client.search(
                [vec],
                self.milvus_vector_field,
                search_params,
                top_k,
                output_fields=self.milvus_output_fields,
            )

        # hits is a list (one per query); take first
        results = []
        if len(hits) == 0:
            return results
        for hit in hits[0]:
            # hit has id and distance/score
            try:
                hit_id = int(hit.id)
            except Exception:
                hit_id = getattr(hit, "id", None)
            score = getattr(hit, "score", None)
            if score is None:
                # pymilvus may call it distance
                score = getattr(hit, "distance", None)
            # collect output fields if present
            out = {
                "idx": hit_id,
                "id": hit_id,
                "score": float(score) if score is not None else None,
            }
            try:
                entity = getattr(hit, "entity", None)
                if entity and isinstance(entity, dict):
                    for k, v in entity.items():
                        out[k] = v
                elif entity is not None:
                    # sometimes entity has get() method
                    for field in self.milvus_output_fields:
                        try:
                            out[field] = entity.get(field)
                        except Exception:
                            out[field] = None
            except Exception:
                pass
            results.append(out)
        return results


if __name__ == "__main__":
    import argparse

    from src.config import load_config

    p = argparse.ArgumentParser(description="Query inference pipeline")
    p.add_argument("query", nargs="+", help="Query text")
    p.add_argument("--top_k", type=int, default=5, help="Number of results")
    p.add_argument("--config", type=str, default=None, help="Path to config file")
    args = p.parse_args()

    query_text = " ".join(args.query)

    try:
        config = load_config("inference", args.config)
        print(f"✅ Loaded config: {config.name}")

        pipe = InferencePipeline(config)
        res = pipe.query(query_text, top_k=args.top_k)

        print(f"\n🔍 Results for query: '{query_text}'\n")
        for i, r in enumerate(res, 1):
            print(f"{i}. [{r['id']}] (score={r['score']:.4f})")
            if r.get("combined_text"):
                print(f"   {r['combined_text'][:200]}...\n")
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        import traceback

        traceback.print_exc()

# Usage:
# python src/inference_pipeline/pipeline.py "sit up" --top_k 5
# python src/inference_pipeline/pipeline.py "push up" --config configs/inference.yaml
# python src/inference_pipeline/pipeline.py "bicep curl" --config configs/inference_prod.yaml --top_k 10
