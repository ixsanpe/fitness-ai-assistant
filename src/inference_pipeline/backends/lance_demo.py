"""Proof-of-concept Lance/LanceDB backend, for comparing against MilvusBackend.

This is a standalone demo, NOT wired into pipeline.py or configs/inference.yaml.
It implements the same SearchBackend interface as MilvusBackend (see milvus.py)
so the two can be compared side by side, and it prints out what Lance actually
writes to disk so the storage model is visible instead of hidden behind a client.

Requires the optional "lance-demo" extra (kept out of the default install
since this is demo-only, not wired into the pipeline):
    uv sync --extra lance-demo
    uv run python -m src.inference_pipeline.backends.lance_demo

Underneath, LanceDB is a thin Python API over the "Lance" file format: a
versioned, columnar format (Arrow-based, similar spirit to Parquet) with:
  - immutable data fragments (*.lance files) written on every insert/append
  - a transaction manifest (_versions/, _transactions/) giving snapshot
    versioning and time travel for free
  - an optional secondary index (_indices/) for ANN search (IVF_PQ / HNSW),
    built explicitly via create_index() once there's enough data to partition
There is no server process: reads/writes go straight to these files (local
disk, or S3/GCS/Azure — same code path). Milvus Lite, by contrast, embeds a
single-file SQLite-backed store behind the same client API Milvus server uses,
so switching to a real Milvus deployment means talking to a server over gRPC
instead of opening a local file.
"""

from pathlib import Path

import numpy as np

from src.inference_pipeline.backends.base import SearchBackend

# Below this row count, an IVF ANN index has too few points to partition
# meaningfully (Lance and Milvus both fall back to brute-force/FLAT search
# on small collections) so we skip create_index() and just do a flat scan.
IVF_INDEX_MIN_ROWS = 256


class LanceBackend(SearchBackend):
    """Vector search using a local Lance dataset via LanceDB."""

    def __init__(
        self,
        db_path: Path | str,
        table_name: str,
        metric: str = "cosine",
        output_fields: list[str] | None = None,
    ):
        """Initialize Lance backend.

        Args:
            db_path: Directory holding the Lance dataset (created if missing)
            table_name: Name of the Lance table to search
            metric: Similarity metric (cosine, l2, dot)
            output_fields: Additional columns to retrieve
        """
        import lancedb

        self.db_path = db_path
        self.table_name = table_name
        self.metric = metric
        self.output_fields = output_fields or []

        self.db = lancedb.connect(str(db_path))
        self.table = self.db.open_table(table_name)
        print(f"✅ Connected to Lance table: {table_name} ({self.table.count_rows()} rows)")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search the Lance table.

        Args:
            query_vector: (D,) query embedding
            top_k: Number of results

        Returns:
            List of result dicts
        """
        vec = query_vector.astype(np.float32)
        print(f"🔍 Querying Lance with vector (len={len(vec)}, first 5={vec[:5].tolist()})")

        # search() is typed to return the base LanceQueryBuilder, but passing
        # a vector makes it a LanceVectorQueryBuilder at runtime, which is
        # where .metric() actually lives — a stub gap in lancedb, not a bug.
        hits = self.table.search(vec).metric(self.metric).limit(top_k).to_list()  # type: ignore[attr-defined]

        results = []
        for hit in hits:
            result = {
                "idx": hit.get("id"),
                "id": hit.get("id"),
                # LanceDB returns a distance (lower = closer); Milvus's client
                # returns a "score" whose direction depends on metric_type.
                # Callers comparing the two need to normalize this themselves.
                "score": float(hit["_distance"]) if "_distance" in hit else None,
            }
            for field in self.output_fields:
                result[field] = hit.get(field)
            results.append(result)

        return results

    def close(self) -> None:
        """No persistent connection to close — every call opens/reads local files."""


def build_lance_table(
    db_path: Path | str,
    table_name: str,
    vectors: np.ndarray,
    items: list[dict],
) -> None:
    """Create (or overwrite) a Lance table from embeddings + metadata.

    Mirrors create_db.py's create_vector_db(), but for Lance: instead of one
    client.insert() call into a running collection, this writes an Arrow
    table straight to a fragment file under db_path/table_name.lance/.
    """
    import lancedb

    db = lancedb.connect(str(db_path))
    data = [
        {
            "id": i,
            "vector": vectors[i].tolist(),
            "text": items[i].get("combined_text", "")[:200],
            "name": items[i].get("id", f"exercise_{i}"),
        }
        for i in range(len(vectors))
    ]

    print(f"Writing {len(data)} rows to Lance table '{table_name}' at {db_path}")
    # mode="overwrite" only makes sense once a table exists; on a fresh path
    # it makes Lance try to open a dataset that isn't there yet and log a
    # (harmless) WARN before creating it. Use "create" the first time round.
    mode = "overwrite" if table_name in db.list_tables().tables else "create"
    table = db.create_table(table_name, data=data, mode=mode)

    if len(vectors) >= IVF_INDEX_MIN_ROWS:
        print(f"Building IVF_PQ index ({len(vectors)} rows >= {IVF_INDEX_MIN_ROWS})...")
        from lancedb.index import IvfPq

        table.create_index("vector", config=IvfPq(distance_type="cosine"))
    else:
        print(
            f"Skipping ANN index: only {len(vectors)} rows "
            f"(< {IVF_INDEX_MIN_ROWS}) — search will be a flat/brute-force scan, "
            "same as Milvus would do on a collection this small."
        )


def inspect_on_disk_layout(db_path: Path) -> None:
    """Print every file Lance wrote, to make the storage model concrete.

    Contrast with Milvus Lite, which keeps everything in one opaque .db file.
    """
    db_path = Path(db_path)
    print(f"\n📁 On-disk layout under {db_path}/")
    total_bytes = 0
    for p in sorted(db_path.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            total_bytes += size
            print(f"   {p.relative_to(db_path)}  ({size:,} bytes)")
    print(f"   Total: {total_bytes:,} bytes")


def main() -> None:
    """Build a Lance table from this project's real exercise embeddings, query
    it, and show what landed on disk — the Lance equivalent of create_db.py.
    """
    from src.inference_pipeline.loaders import load_embeddings, load_metadata

    project_root = Path(__file__).resolve().parents[3]
    embeddings_dir = project_root / "data/processed/embeddings"
    dataset_path = project_root / "data/processed/exercises_dataset.jsonl"
    db_path = project_root / "data/vector_db/lance_demo.db"

    vectors = load_embeddings(embeddings_dir, "sentence")
    items = load_metadata(dataset_path)
    n = min(len(vectors), len(items))
    vectors, items = vectors[:n], items[:n]

    build_lance_table(db_path, "exercises_embeddings", vectors, items)
    inspect_on_disk_layout(db_path)

    backend = LanceBackend(
        db_path,
        "exercises_embeddings",
        metric="cosine",
        output_fields=["name", "text"],
    )
    query_vector = vectors[0]  # stand-in for TextEmbedder.embed(text)
    results = backend.search(query_vector, top_k=5)

    print("\n🏋️  Top 5 matches:")
    for r in results:
        print(f"   {r['score']:.4f}  {r.get('name')}")

    backend.close()


if __name__ == "__main__":
    main()
