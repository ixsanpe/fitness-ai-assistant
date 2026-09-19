"""Retrieval metrics (precision/recall/AP/MRR/nDCG@k) and the evaluation
harness that runs a retrieve_fn over a labeled test set and writes a
markdown + PNG report.

Usage:
    python -m src.eval.eval_harness --testset data/eval/test_queries.jsonl --top_k 5
    python -m src.eval.eval_harness --config configs/inference.yaml
"""

import json
import math
from collections.abc import Callable
from pathlib import Path

from src.config import InferenceConfig, load_config
from src.inference_pipeline.pipeline import InferencePipeline

# A retrieve_fn takes a query string and top_k, and returns a ranked list of
# item ids — the common interface dense backends, BM25, and RRF fusion all
# satisfy, so the same metric code can score any of them.
RetrieveFn = Callable[[str, int], list[str]]


def precision_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(1 for r in retrieved_k if r in set(relevant))
    return hits / float(k)


def recall_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in set(relevant))
    return hits / float(len(relevant))


def average_precision_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    # computes AP@k
    retrieved_k = retrieved[:k]
    score = 0.0
    num_hits = 0
    for i, r in enumerate(retrieved_k, start=1):
        if r in set(relevant):
            num_hits += 1
            score += num_hits / float(i)
    if num_hits == 0:
        return 0.0
    return score / float(min(len(relevant), k))


def _reciprocal_rank(rel: list[int]) -> float:
    for i, v in enumerate(rel, start=1):
        if v:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(relevances: list[list[int]]) -> float:
    # relevances: for each query a list of binary rels in retrieved order
    if not relevances:
        return 0.0
    return sum(_reciprocal_rank(rel) for rel in relevances) / len(relevances)


def dcg(relevances: list[int], k: int) -> float:
    dcg_v = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg_v += (2**rel - 1) / float(math.log2(i + 1))
    return dcg_v


def ndcg_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    # treat relevance as binary
    rels = [1 if r in set(relevant) else 0 for r in retrieved[:k]]
    if not any(rels):
        return 0.0
    # DCG
    dcg_v = 0.0
    for i, rel in enumerate(rels, start=1):
        dcg_v += rel / float(math.log2(i + 1))
    # IDCG
    ideal = sorted(rels, reverse=True)
    idcg_v = 0.0
    for i, rel in enumerate(ideal, start=1):
        idcg_v += rel / float(math.log2(i + 1))
    return dcg_v / idcg_v if idcg_v > 0 else 0.0


def load_testset(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def evaluate_retriever(testset: list[dict], retrieve_fn: RetrieveFn, top_k: int = 5) -> dict:
    """Score any retriever against a loaded testset.

    Args:
        testset: list of {"query": str, "relevant": list[str]} dicts
        retrieve_fn: query, top_k -> ranked list of item ids
        top_k: cutoff for all metrics

    Returns a dict of aggregate metrics plus a "per_query" breakdown (one
    entry per testset row) for spotting which queries the retriever struggles
    with, rather than only the corpus-wide average.
    """
    precisions = []
    recalls = []
    aps = []
    ndcgs = []
    relevances_for_mrr = []
    per_query = []

    for t in testset:
        q = t.get("query")
        relevant = t.get("relevant", [])
        retrieved = [str(rid) for rid in retrieve_fn(q, top_k)]

        p = precision_at_k(relevant, retrieved, top_k)
        r = recall_at_k(relevant, retrieved, top_k)
        ap = average_precision_at_k(relevant, retrieved, top_k)
        ndcg = ndcg_at_k(relevant, retrieved, top_k)
        rel_binary = [1 if rid in set(relevant) else 0 for rid in retrieved]

        precisions.append(p)
        recalls.append(r)
        aps.append(ap)
        ndcgs.append(ndcg)
        relevances_for_mrr.append(rel_binary)
        per_query.append(
            {
                "query": q,
                "num_relevant": len(relevant),
                "num_retrieved": len(retrieved),
                "precision@k": p,
                "recall@k": r,
                "ap@k": ap,
                "ndcg@k": ndcg,
                "rr": _reciprocal_rank(rel_binary),
            }
        )

    mrr = mean_reciprocal_rank(relevances_for_mrr)
    return {
        "precision@k": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall@k": sum(recalls) / len(recalls) if recalls else 0.0,
        "map@k": sum(aps) / len(aps) if aps else 0.0,
        "ndcg@k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "mrr": mrr,
        "num_queries": len(testset),
        "k": top_k,
        "per_query": per_query,
    }


def evaluate(testset_path: str, config: InferenceConfig = None, top_k: int = 5) -> dict:
    """Run evaluation of the configured InferencePipeline on a JSONL testset.

    Testset format (per-line JSON):
      {"query": "sit up", "relevant": ["Otis-Up", "3_4_Sit-Up"]}

    Returns a dict of aggregate metrics. For scoring other retrievers (BM25,
    hybrid fusion, a raw backend, ...) against the same testset, use
    `evaluate_retriever()` directly with a `retrieve_fn`.
    """
    if config is None:
        config = load_config("inference", "configs/inference.yaml")

    pipe = InferencePipeline(config)

    path = Path(testset_path)
    if not path.exists():
        raise FileNotFoundError(f"Testset not found: {path}")

    testset = load_testset(path)

    def retrieve_fn(query: str, k: int) -> list[str]:
        results = pipe.query(query, top_k=k)
        # "name" carries the true exercise id for backends (e.g. Milvus) whose
        # "id" is just an internal row PK; other backends don't set "name" and
        # already put the exercise id in "id" (see generation.py for the same
        # precedence).
        return [str(r.get("name") or r.get("id") or r.get("idx")) for r in results]

    return evaluate_retriever(testset, retrieve_fn, top_k=top_k)


def _write_report(metrics: dict, output_md: Path) -> None:
    per_query = metrics["per_query"]
    lines = [
        "| Metric | Value |",
        "|---|---:|",
        f"| precision@{metrics['k']} | {metrics['precision@k']:.3f} |",
        f"| recall@{metrics['k']} | {metrics['recall@k']:.3f} |",
        f"| map@{metrics['k']} | {metrics['map@k']:.3f} |",
        f"| ndcg@{metrics['k']} | {metrics['ndcg@k']:.3f} |",
        f"| mrr | {metrics['mrr']:.3f} |",
        f"| num_queries | {metrics['num_queries']} |",
        "",
        f"Worst {min(10, len(per_query))} queries by ndcg@{metrics['k']}:",
        "",
        "| Query | #relevant | precision | recall | ndcg |",
        "|---|---:|---:|---:|---:|",
    ]
    worst = sorted(per_query, key=lambda r: r["ndcg@k"])[:10]
    for r in worst:
        lines.append(
            f"| {r['query']} | {r['num_relevant']} | {r['precision@k']:.3f} | "
            f"{r['recall@k']:.3f} | {r['ndcg@k']:.3f} |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_md}")


def _plot_report(metrics: dict, output_png: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    per_query = metrics["per_query"]
    k = metrics["k"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Aggregate metrics at a glance
    agg_labels = [f"precision@{k}", f"recall@{k}", f"map@{k}", f"ndcg@{k}", "mrr"]
    agg_values = [metrics[key] for key in ("precision@k", "recall@k", "map@k", "ndcg@k", "mrr")]
    axes[0].bar(agg_labels, agg_values, color="#4C72B0")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Aggregate metrics")
    axes[0].tick_params(axis="x", rotation=30)

    # Distribution of per-query ndcg — shows whether the retriever is
    # uniformly mediocre or bimodal (great on some queries, failing on others)
    ndcgs = [r["ndcg@k"] for r in per_query]
    axes[1].hist(ndcgs, bins=15, range=(0, 1), color="#55A868", edgecolor="white")
    axes[1].set_xlabel(f"ndcg@{k}")
    axes[1].set_ylabel("number of queries")
    axes[1].set_title("Per-query ndcg distribution")

    # recall@k is capped by how many relevant items exist relative to k, so
    # plotting it against num_relevant shows how much of "low recall" is
    # structural rather than a retrieval failure
    num_relevant = [r["num_relevant"] for r in per_query]
    recalls = [r["recall@k"] for r in per_query]
    axes[2].scatter(num_relevant, recalls, alpha=0.6, color="#C44E52")
    axes[2].axvline(k, color="gray", linestyle="--", linewidth=1, label=f"k={k}")
    axes[2].set_xlabel("# relevant items for query")
    axes[2].set_ylabel(f"recall@{k}")
    axes[2].set_title("Recall vs. relevant-set size")
    axes[2].legend()

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Evaluate inference pipeline on test queries")
    p.add_argument(
        "--testset", default="data/eval/test_queries.jsonl", help="Path to test queries JSONL"
    )
    p.add_argument("--top_k", type=int, default=5, help="Number of results to retrieve")
    p.add_argument(
        "--config", type=str, default="configs/inference.yaml", help="Path to inference config file"
    )
    p.add_argument("--output_json", default="results/eval_harness.json")
    p.add_argument("--output_md", default="results/eval_harness.md")
    p.add_argument("--output_png", default="results/eval_harness.png")
    args = p.parse_args()

    config = load_config("inference", args.config)
    metrics = evaluate(args.testset, config=config, top_k=args.top_k)

    print("\n📊 Evaluation Results:")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_query"}, indent=2))

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nWrote {output_json}")

    _write_report(metrics, Path(args.output_md))
    _plot_report(metrics, Path(args.output_png))
