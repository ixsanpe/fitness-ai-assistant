"""Retrieval quality benchmark: BM25 vs dense (sentence/CLIP) vs hybrid RRF.

Runs every retriever over the same labeled eval set
(`data/eval/test_queries.jsonl`, see `build_eval_set.py`) and reports
recall@k / nDCG@k / MRR for each, so the embedding-model comparison this
project already does has actual numbers behind it instead of just an
architecture diagram.

Usage:
    python -m src.eval.benchmark_retrieval --top_k 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.eval_harness import RetrieveFn, evaluate_retriever, load_testset
from src.eval.fusion import reciprocal_rank_fusion
from src.inference_pipeline.backends import BM25Retriever, LocalBackend
from src.inference_pipeline.loaders import load_embeddings, load_metadata
from src.inference_pipeline.models import TextEmbedder

DEFAULT_RRF_POOL_SIZE = 20


def build_bm25_retrieve_fn(metadata: list[dict]) -> RetrieveFn:
    retriever = BM25Retriever(metadata)
    return retriever.retrieve


def build_sentence_retrieve_fn(
    model_name_or_path: str,
    embeddings: np.ndarray,
    metadata: list[dict],
    device: str = "cpu",
) -> RetrieveFn:
    embedder = TextEmbedder(model_name_or_path, device=device)
    backend = LocalBackend(embeddings, metadata)

    def retrieve(query: str, top_k: int) -> list[str]:
        vec = embedder.embed(query)
        return [str(r["id"]) for r in backend.search(vec, top_k=top_k)]

    return retrieve


def embed_clip_query(
    text: str,
    model,
    processor,
    text_weight: float,
    image_weight: float,
    projection_dim: int,
    normalize: bool,
) -> np.ndarray:
    """Embed a text-only query into the same [text | image] concatenated
    space `CLIPEmbedder` builds the corpus in (`_combine_embeddings` in
    `src/feature_pipeline/embedders/clip.py`), padding the missing image half
    with zeros — the same convention that embedder already uses for exercises
    with no images."""
    text_input = processor.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_output = model.get_text_features(**text_input)
        if isinstance(text_output, torch.Tensor):
            text_emb = text_output
        else:
            text_emb = model.text_projection(text_output.pooler_output)
    image_emb = torch.zeros((1, projection_dim))
    combined = torch.cat([text_emb * text_weight, image_emb * image_weight], dim=1)
    if normalize:
        combined = torch.nn.functional.normalize(combined, p=2, dim=1)
    return combined[0].numpy().astype(np.float32)


def build_clip_retrieve_fn(
    model_name: str,
    embeddings: np.ndarray,
    metadata: list[dict],
    text_weight: float = 0.5,
    image_weight: float = 0.5,
    normalize: bool = True,
) -> RetrieveFn:
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
    projection_dim = model.config.projection_dim
    backend = LocalBackend(embeddings, metadata)

    def retrieve(query: str, top_k: int) -> list[str]:
        vec = embed_clip_query(
            query, model, processor, text_weight, image_weight, projection_dim, normalize
        )
        return [str(r["id"]) for r in backend.search(vec, top_k=top_k)]

    return retrieve


def build_hybrid_retrieve_fn(
    fn_a: RetrieveFn, fn_b: RetrieveFn, pool_size: int = DEFAULT_RRF_POOL_SIZE
) -> RetrieveFn:
    """Fuse two retrievers with Reciprocal Rank Fusion.

    Each retriever is queried for `pool_size` candidates (deeper than the
    eventual top_k) before fusing, so RRF has enough ranked signal to work
    with rather than truncating each side to top_k first.
    """

    def retrieve(query: str, top_k: int) -> list[str]:
        pool = max(pool_size, top_k)
        ranking_a = fn_a(query, pool)
        ranking_b = fn_b(query, pool)
        fused = reciprocal_rank_fusion([ranking_a, ranking_b])
        return fused[:top_k]

    return retrieve


def run_benchmark(
    systems: dict[str, RetrieveFn], testset: list[dict], top_k: int
) -> dict[str, dict]:
    results = {}
    for name, retrieve_fn in systems.items():
        print(f"Evaluating: {name}")
        results[name] = evaluate_retriever(testset, retrieve_fn, top_k=top_k)
    return results


def format_markdown_table(results: dict[str, dict], top_k: int) -> str:
    lines = [
        f"| System | Recall@{top_k} | nDCG@{top_k} | MRR | MAP@{top_k} |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in results.items():
        lines.append(
            f"| {name} | {metrics['recall@k']:.3f} | {metrics['ndcg@k']:.3f} | "
            f"{metrics['mrr']:.3f} | {metrics['map@k']:.3f} |"
        )
    return "\n".join(lines)


def build_default_systems(
    metadata: list[dict],
    sentence_embeddings: np.ndarray,
    clip_embeddings: np.ndarray,
    sentence_model_name: str,
    clip_model_name: str,
) -> dict[str, RetrieveFn]:
    bm25_fn = build_bm25_retrieve_fn(metadata)
    sentence_fn = build_sentence_retrieve_fn(sentence_model_name, sentence_embeddings, metadata)
    clip_fn = build_clip_retrieve_fn(clip_model_name, clip_embeddings, metadata)

    return {
        "BM25": bm25_fn,
        "Sentence-Transformers (dense)": sentence_fn,
        "CLIP text (dense)": clip_fn,
        "Hybrid RRF (BM25 + Sentence)": build_hybrid_retrieve_fn(bm25_fn, sentence_fn),
        "Hybrid RRF (BM25 + CLIP)": build_hybrid_retrieve_fn(bm25_fn, clip_fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", default="data/eval/test_queries.jsonl")
    parser.add_argument("--dataset", default="data/processed/exercises_dataset.jsonl")
    parser.add_argument("--embeddings_dir", default="data/processed/embeddings")
    parser.add_argument("--sentence_model", default="all-MiniLM-L6-v2")
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_json", default="results/retrieval_benchmark.json")
    parser.add_argument("--output_md", default="results/retrieval_benchmark.md")
    args = parser.parse_args()

    metadata = load_metadata(Path(args.dataset))
    sentence_embeddings = load_embeddings(Path(args.embeddings_dir), "sentence")
    clip_embeddings = load_embeddings(Path(args.embeddings_dir), "clip")

    testset = load_testset(Path(args.testset))
    print(f"Loaded {len(testset)} eval queries from {args.testset}")

    systems = build_default_systems(
        metadata, sentence_embeddings, clip_embeddings, args.sentence_model, args.clip_model
    )
    results = run_benchmark(systems, testset, top_k=args.top_k)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {output_json}")

    table = format_markdown_table(results, args.top_k)
    output_md = Path(args.output_md)
    output_md.write_text(table + "\n", encoding="utf-8")
    print(f"Wrote {output_md}")
    print("\n" + table)


if __name__ == "__main__":
    main()
