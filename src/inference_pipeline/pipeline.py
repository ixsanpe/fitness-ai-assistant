from pathlib import Path
import numpy as np
import json
from typing import List, Dict, Optional

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
try:
    # pymilvus is optional; if present we can use a Milvus-backed search
    from pymilvus import connections, Collection
except Exception:
    connections = None
    Collection = None


EMBEDDINGS_DIR = Path("data/processed/embeddings")
DEFAULT_DATASET_PATH = Path("data/processed/exercises_dataset.jsonl")


def find_embeddings_file(embedding_type: str = "sentence") -> Optional[Path]:
    """Find an embeddings file in EMBEDDINGS_DIR.

    embedding_type: "sentence" -> text embeddings (endswith text.npy)
                    "clip" -> clip embeddings (endswith clip.npy)
    """
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if embedding_type == "sentence":
        pat = "*text.npy"
    elif embedding_type == "clip":
        pat = "*clip.npy"
    else:
        raise ValueError(f"Unknown embedding_type {embedding_type}")
    files = list(EMBEDDINGS_DIR.glob(pat))
    return files[0] if files else None


def load_embeddings(embedding_type: str = "sentence") -> np.ndarray:
    path = find_embeddings_file(embedding_type)
    if path is None:
        raise FileNotFoundError(f"No embeddings file found for type={embedding_type} in {EMBEDDINGS_DIR}")
    arr = np.load(str(path))
    # ensure 2D
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    print(f"Loaded embeddings from {path}, shape={arr.shape}")
    return arr


def load_metadata(dataset_path: Path = DEFAULT_DATASET_PATH) -> List[Dict]:
    items = []
    with open(dataset_path, "r", encoding="utf-8") as f:
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

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_type: str = "sentence",
        device: str = "cpu",
        backend: str = "milvus",  # or 'milvus'
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        milvus_collection: str = "exercises_embeddings",
        milvus_vector_field: str = "vector",
        milvus_output_fields: Optional[List[str]] = None,
    ):
        self.embedding_type = embedding_type
        self.device = device
        self.model_name = model_name
        self.backend = backend

        # load numpy embeddings and metadata for local backend
        self.embeddings = None
        self.metadata = None
        if backend == "local":
            self.embeddings = load_embeddings(embedding_type)
            self.metadata = load_metadata()
            if len(self.metadata) != self.embeddings.shape[0]:
                # still allow but warn (we don't have logging configured - use print)
                print(f"Warning: metadata count {len(self.metadata)} != embeddings rows {self.embeddings.shape[0]}")

            # normalize embeddings for cosine similarity
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self._embeddings_normalized = self.embeddings / norms

        # milvus params (optional)
        self.milvus_db_name = "milvus_demo.db"
        self.milvus_collection_name = milvus_collection
        self.milvus_vector_field = milvus_vector_field
        self.milvus_output_fields = milvus_output_fields or []
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

        # model can be lazy-loaded to avoid heavy import at init
        self._model = None

        # lazy-initialize milvus collection handle if requested
        self._milvus_collection = None
        if backend == "milvus":
            self.milvus_client = MilvusClient(self.milvus_db_name)
            collection_info = self.milvus_client.describe_collection(collection_name="exercises_embeddings")
            print(f"Collection info: {collection_info}")

    def _ensure_model(self):
        if self._model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not installed; install with `pip install sentence-transformers`")
            self._model = SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> np.ndarray:
        """Return a single normalized embedding for the input text."""
        self._ensure_model()
        emb = self._model.encode([text], convert_to_numpy=True)[0].astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
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
        elif self.backend == "milvus":
            return self._query_milvus(q, top_k=top_k)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


    def _query_milvus(self, q: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search Milvus collection and return results translated to our result format."""
        # ensure query vector is python list
        vec = q.astype(np.float32).tolist()
        print(f"First few values: {vec[:5]}")  # Check it's actual numbers
        # build search params — for cosine we use COSINE on normalized vectors
        search_params = {"metric_type": "COSINE", "params": {}}
        # perform search
        try:
            print(f" The len of the vector is {len(vec)}")
            print(f" Other params are: {self.milvus_vector_field}, {search_params}, {top_k}, {self.milvus_output_fields}")
            #hits = self.milvus_client.search([vec], anns_field=self.milvus_vector_field, param=search_params, limit=top_k, output_fields=self.milvus_output_fields)
            
            hits = self.milvus_client.search(
                collection_name="exercises_embeddings",
                data=[vec],  # List containing your vector
                anns_field=self.milvus_vector_field,
                limit=top_k,
                output_fields=self.milvus_output_fields,
                search_params=search_params
            )
            
        except TypeError:
            # some pymilvus versions name the arg `param` differently; try without named args
            print("There was a TypeError, trying alternative search call...")
            hits = self.milvus_client.search([vec], self.milvus_vector_field, search_params, top_k, output_fields=self.milvus_output_fields)

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
            out = {"idx": hit_id, "id": hit_id, "score": float(score) if score is not None else None}
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

    p = argparse.ArgumentParser()
    p.add_argument("--query", nargs="+", help="Query text")
    p.add_argument("--top_k", type=int, default=5)
    args = p.parse_args()
    query_text = " ".join(args.query)
    try:
        pipe = InferencePipeline()
        res = pipe.query(query_text, top_k=args.top_k)
        print(f"Results for query: '{res}'")
        for r in res:
            print(f"[{r['idx']}] {r['id']} (score={r['score']:.4f})\n  {r['combined_text'][:200]}...\n")
    except Exception as e:
        print("Error running pipeline:", e)


    #TODO: improve selection of embeddings file (sentence, clip), plus the db, plus the pipeline... better model selection