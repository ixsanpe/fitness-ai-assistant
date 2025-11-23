import json
import math
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import InferenceConfig, load_config
from src.inference_pipeline.pipeline import InferencePipeline


def precision_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(1 for r in retrieved_k if r in set(relevant))
    return hits / float(k)


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


def mean_reciprocal_rank(relevances: list[list[int]]) -> float:
    # relevances: for each query a list of binary rels in retrieved order
    rr_total = 0.0
    for rel in relevances:
        rr = 0.0
        for i, v in enumerate(rel, start=1):
            if v:
                rr = 1.0 / i
                break
        rr_total += rr
    return rr_total / len(relevances) if relevances else 0.0


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


def evaluate(testset_path: str, config: InferenceConfig = None, top_k: int = 5) -> dict:
    """Run evaluation on a JSONL testset.

    Testset format (per-line JSON):
      {"query": "sit up", "relevant": ["Otis-Up", "3_4_Sit-Up"]}

    Returns a dict of aggregate metrics.
    """
    if config is None:
        config = load_config("inference", "configs/inference.yaml")

    pipe = InferencePipeline(config)

    path = Path(testset_path)
    if not path.exists():
        raise FileNotFoundError(f"Testset not found: {path}")

    tests = load_testset(path)
    precisions = []
    aps = []
    ndcgs = []
    relevances_for_mrr = []

    for t in tests:
        q = t.get("query")
        relevant = t.get("relevant", [])
        results = pipe.query(q, top_k=top_k)
        # extract returned ids (string ids preferred; fallback to idx)
        retrieved = []
        for r in results:
            rid = r.get("id") if r.get("id") is not None else r.get("idx")
            retrieved.append(str(rid))

        p = precision_at_k(relevant, retrieved, top_k)
        precisions.append(p)
        ap = average_precision_at_k(relevant, retrieved, top_k)
        aps.append(ap)
        ndcg_v = ndcg_at_k(relevant, retrieved, top_k)
        ndcgs.append(ndcg_v)
        # binary rel list for MRR
        relevances_for_mrr.append([1 if rid in set(relevant) else 0 for rid in retrieved])

    mrr = mean_reciprocal_rank(relevances_for_mrr)
    metrics = {
        "precision@k": sum(precisions) / len(precisions) if precisions else 0.0,
        "map@k": sum(aps) / len(aps) if aps else 0.0,
        "ndcg@k": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
        "mrr": mrr,
        "num_queries": len(tests),
        "k": top_k,
    }
    return metrics


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
    args = p.parse_args()

    config = load_config("inference", args.config)
    metrics = evaluate(args.testset, config=config, top_k=args.top_k)

    print("\n📊 Evaluation Results:")
    print(json.dumps(metrics, indent=2))

# Usage:
# python src/eval/eval_harness.py --testset data/eval/test_queries.jsonl --top_k 5
# python src/eval/eval_harness.py --config configs/inference.yaml
# python src/eval/eval_harness.py --config configs/inference_prod.yaml --top_k 10
