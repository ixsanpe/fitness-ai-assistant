"""ANN index sweep on Milvus Lite: FLAT vs IVF_FLAT vs HNSW, recall vs p95 latency.

Milvus Lite's local mode rejects an explicit `index_type="HNSW"` at
`create_index` time ("local mode only support FLAT IVF_FLAT AUTOINDEX").
`index_type="AUTOINDEX"` is the only way to get a graph index locally, and
`describe_index()` confirms it actually builds HNSW under the hood (its
resolved params are `M`/`efConstruction`, the HNSW build parameters) — so
that's what "HNSW" means below: `AUTOINDEX` with explicit `M`/`efConstruction`,
queried with a custom `ef`. This is a real constraint of the embedded/local
build, not of Milvus in general — a full Milvus server accepts `HNSW`
directly.

FLAT is exact search, so it's used as ground truth: for every other config,
recall@k here means "how often does this config's top-k match FLAT's top-k",
not agreement with the labeled eval set (that's what `benchmark_retrieval.py`
measures). This isolates the ANN approximation's effect from embedding
quality.

Usage:
    python -m src.eval.index_sweep --top_k 10 --num_trials 200
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pymilvus import DataType, MilvusClient

from src.eval.eval_harness import load_testset
from src.inference_pipeline.loaders import load_embeddings, load_metadata
from src.inference_pipeline.models import TextEmbedder

# (label, index_type, build_params, search_params)
#
# At this corpus's size (873 rows, 384-dim) every config below recovers ~exact
# search — the "moderate" settings (nprobe/ef sized the way you actually would
# in production, scanning a healthy fraction of the index) all land at
# recall@10=1.0 and sub-millisecond latency, indistinguishable from FLAT. The
# "starved" configs deliberately under-scan (nprobe=1 out of 128 partitions;
# ef=1) to force real approximation error, so the recall/latency tradeoff this
# sweep exists to demonstrate is actually visible instead of flatlining.
SWEEP_CONFIGS = [
    ("FLAT", "FLAT", {}, {}),
    ("IVF_FLAT (nlist=16, nprobe=4)", "IVF_FLAT", {"nlist": 16}, {"nprobe": 4}),
    ("IVF_FLAT (nlist=64, nprobe=8)", "IVF_FLAT", {"nlist": 64}, {"nprobe": 8}),
    (
        "IVF_FLAT starved (nlist=128, nprobe=1)",
        "IVF_FLAT",
        {"nlist": 128},
        {"nprobe": 1},
    ),
    (
        "HNSW (M=16, efConstruction=128, ef=32)",
        "AUTOINDEX",
        {"M": 16, "efConstruction": 128},
        {"ef": 32},
    ),
    (
        "HNSW (M=32, efConstruction=256, ef=64)",
        "AUTOINDEX",
        {"M": 32, "efConstruction": 256},
        {"ef": 64},
    ),
    (
        "HNSW starved (M=4, efConstruction=8, ef=1)",
        "AUTOINDEX",
        {"M": 4, "efConstruction": 8},
        {"ef": 1},
    ),
]


def build_collection(
    client: MilvusClient,
    name: str,
    dim: int,
    index_type: str,
    build_params: dict,
    vectors: np.ndarray,
    ids: list[str],
) -> None:
    if client.has_collection(collection_name=name):
        client.drop_collection(collection_name=name)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("ext_id", DataType.VARCHAR, max_length=256)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector", index_type=index_type, metric_type="COSINE", params=build_params
    )
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)

    data = [{"vector": vectors[i].tolist(), "ext_id": ids[i]} for i in range(len(ids))]
    client.insert(collection_name=name, data=data)


def search(
    client: MilvusClient, name: str, vector: np.ndarray, top_k: int, search_params: dict
) -> tuple[list[str], float]:
    start = time.perf_counter()
    hits = client.search(
        collection_name=name,
        data=[vector.tolist()],
        anns_field="vector",
        limit=top_k,
        search_params={"metric_type": "COSINE", "params": search_params},
        output_fields=["ext_id"],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    ids = [h.get("entity", {}).get("ext_id") for h in hits[0]]
    return ids, elapsed_ms


def recall_vs_ground_truth(retrieved: list[str], ground_truth: list[str]) -> float:
    if not ground_truth:
        return 0.0
    hits = sum(1 for r in retrieved if r in set(ground_truth))
    return hits / float(len(ground_truth))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/processed/exercises_dataset.jsonl")
    parser.add_argument("--embeddings_dir", default="data/processed/embeddings")
    parser.add_argument("--testset", default="data/eval/test_queries.jsonl")
    parser.add_argument("--sentence_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--num_trials",
        type=int,
        default=200,
        help="Query trials (sampled with replacement from"
        " the eval set, padded up since 50-100 labeled queries is too few for a stable p95)",
    )
    parser.add_argument("--db_dir", default="data/vector_db/index_sweep")
    parser.add_argument("--output_json", default="results/index_sweep.json")
    parser.add_argument("--output_md", default="results/index_sweep.md")
    parser.add_argument("--output_png", default="results/index_sweep.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = load_metadata(Path(args.dataset))
    ids = [str(m["id"]) for m in metadata]
    vectors = load_embeddings(Path(args.embeddings_dir), "sentence")
    dim = vectors.shape[1]

    testset = load_testset(Path(args.testset))
    embedder = TextEmbedder(args.sentence_model, device="cpu")
    query_vectors = [embedder.embed(t["query"]) for t in testset]
    print(f"Embedded {len(query_vectors)} eval queries for the sweep")

    rng = np.random.default_rng(args.seed)
    trial_indices = rng.integers(0, len(query_vectors), size=args.num_trials)

    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    client = MilvusClient(str(db_dir / "sweep.db"))

    ground_truth_by_query: dict[int, list[str]] = {}
    results = {}

    for label, index_type, build_params, search_params in SWEEP_CONFIGS:
        print(f"\nBuilding index: {label}")
        collection_name = "sweep_" + "".join(c if c.isalnum() else "_" for c in label.lower())
        build_collection(client, collection_name, dim, index_type, build_params, vectors, ids)

        latencies_ms = []
        recalls = []
        for i, qi in enumerate(trial_indices):
            retrieved, elapsed_ms = search(
                client, collection_name, query_vectors[qi], args.top_k, search_params
            )
            if i > 0:  # drop the first call as warmup (cold cache)
                latencies_ms.append(elapsed_ms)

            if label == "FLAT":
                ground_truth_by_query[qi] = retrieved
                recalls.append(1.0)
            else:
                recalls.append(recall_vs_ground_truth(retrieved, ground_truth_by_query.get(qi, [])))

        results[label] = {
            "index_type": index_type,
            "build_params": build_params,
            "search_params": search_params,
            f"recall@{args.top_k}_vs_flat": float(np.mean(recalls)),
            "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
            "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
            "num_trials": len(trial_indices),
        }
        print(
            f"  recall@{args.top_k} vs FLAT: {results[label][f'recall@{args.top_k}_vs_flat']:.3f}"
        )
        print(f"  p95 latency: {results[label]['p95_latency_ms']:.2f} ms")

        client.drop_collection(collection_name=collection_name)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {output_json}")

    lines = [
        f"| Index | Recall@{args.top_k} vs FLAT | p50 latency (ms) | p95 latency (ms) |",
        "|---|---:|---:|---:|",
    ]
    for label, r in results.items():
        lines.append(
            f"| {label} | {r[f'recall@{args.top_k}_vs_flat']:.3f} | "
            f"{r['p50_latency_ms']:.3f} | {r['p95_latency_ms']:.3f} |"
        )
    table = "\n".join(lines)
    output_md = Path(args.output_md)
    output_md.write_text(table + "\n", encoding="utf-8")
    print(f"Wrote {output_md}")
    print("\n" + table)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        for label, r in results.items():
            ax.scatter(r["p95_latency_ms"], r[f"recall@{args.top_k}_vs_flat"], s=80, label=label)
        ax.set_xlabel("p95 latency (ms)")
        ax.set_ylabel(f"recall@{args.top_k} vs FLAT")
        # Fixed 0-1.05 range: at this corpus size every config lands at ~1.0
        # recall (see module docstring), and letting matplotlib auto-zoom the
        # y-axis exaggerates noise-level differences into a false signal.
        ax.set_ylim(0, 1.05)
        ax.set_title("Index sweep: recall vs p95 latency (Milvus Lite, local mode)")
        ax.grid(True, alpha=0.3)
        # A legend, not inline annotations: p95 latencies cluster too tightly
        # for per-point labels to avoid overlapping each other.
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, title="index config")
        fig.tight_layout()
        output_png = Path(args.output_png)
        fig.savefig(output_png, dpi=150)
        print(f"Wrote {output_png}")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
