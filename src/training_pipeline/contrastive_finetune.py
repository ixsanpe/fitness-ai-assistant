"""Contrastive fine-tune of the sentence embedder on bootstrapped exercise
query/positive pairs (see `src/eval/build_eval_set.py`), using
MultipleNegativesRankingLoss — the standard "in-batch negatives" contrastive
loss for retrieval: each (query, positive) pair in a batch treats every other
pair's positive as a negative, so no explicit negative mining is needed.

Fills in `training_pipeline/`, and closes the loop on the embedding-model
comparison this project already does: instead of just comparing off-the-shelf
models, this produces a model actually adapted to the exercise domain and
reports the gain (or lack of one) on the same eval set / benchmark harness
everything else in `src/eval/` uses.

Usage:
    python -m src.training_pipeline.contrastive_finetune \
        --config configs/training_contrastive.yaml
"""

import argparse
import json
import re
from pathlib import Path

import mlflow
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

from src.config import TrainingConfig, load_config
from src.config.training_config import LossFunction
from src.eval.benchmark_retrieval import build_sentence_retrieve_fn, format_markdown_table
from src.eval.eval_harness import evaluate_retriever, load_testset
from src.inference_pipeline.loaders import load_embeddings, load_metadata


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def build_training_dataset(pairs: list[dict]) -> Dataset:
    """MultipleNegativesRankingLoss expects columns it can pair up positionally
    — here (anchor, positive) — with no explicit "label" column."""
    return Dataset.from_dict(
        {
            "anchor": [p["query"] for p in pairs],
            "positive": [p["positive_text"] for p in pairs],
        }
    )


def run_training(config: TrainingConfig) -> Path:
    if config.loss_function != LossFunction.MULTIPLE_NEGATIVES_RANKING:
        raise ValueError(
            f"This script only implements multiple_negatives_ranking, got {config.loss_function}"
        )

    pairs = load_pairs(Path(config.data.train_path))
    print(f"Loaded {len(pairs)} contrastive training pairs from {config.data.train_path}")

    dataset = build_training_dataset(pairs)
    split = dataset.train_test_split(test_size=config.data.val_split, seed=config.seed)

    device = config.device.resolve()
    model = SentenceTransformer(config.model_name, device=device)
    loss = MultipleNegativesRankingLoss(model)

    save_dir = Path(config.checkpoint.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(save_dir / "_trainer_output"),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.data.batch_size,
        per_device_eval_batch_size=config.data.eval_batch_size or config.data.batch_size,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        warmup_steps=config.scheduler.num_warmup_steps,
        dataloader_drop_last=config.data.drop_last,
        logging_steps=config.logging_steps,
        eval_strategy=config.evaluation.eval_strategy,
        save_strategy="no",  # we save the final model explicitly below
        seed=config.seed,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        loss=loss,
    )
    trainer.train()

    model.save_pretrained(str(save_dir))
    print(f"Saved fine-tuned model to {save_dir}")
    return save_dir


def compare_before_after(
    config: TrainingConfig,
    finetuned_dir: Path,
    dataset_path: str,
    embeddings_dir: str,
    testset_path: str,
    top_k: int = 10,
) -> dict:
    """Re-embed the corpus with the fine-tuned model and benchmark it against
    the off-the-shelf base model on the same eval set."""
    metadata = load_metadata(Path(dataset_path))
    base_embeddings = load_embeddings(Path(embeddings_dir), "sentence")

    finetuned_model = SentenceTransformer(str(finetuned_dir), device="cpu")
    texts = [m.get("combined_text", "") for m in metadata]
    finetuned_embeddings = finetuned_model.encode(
        texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True
    )

    testset = load_testset(Path(testset_path))

    base_fn = build_sentence_retrieve_fn(config.model_name, base_embeddings, metadata)
    finetuned_fn = build_sentence_retrieve_fn(str(finetuned_dir), finetuned_embeddings, metadata)

    results = {
        "sentence (off-the-shelf)": evaluate_retriever(testset, base_fn, top_k=top_k),
        "sentence (fine-tuned)": evaluate_retriever(testset, finetuned_fn, top_k=top_k),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_contrastive.yaml")
    parser.add_argument("--dataset", default="data/processed/exercises_dataset.jsonl")
    parser.add_argument("--embeddings_dir", default="data/processed/embeddings")
    parser.add_argument("--testset", default="data/eval/test_queries.jsonl")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_json", default="results/contrastive_finetune.json")
    parser.add_argument("--output_md", default="results/contrastive_finetune.md")
    parser.add_argument("--tracking_uri", default="file:./mlruns")
    args = parser.parse_args()

    config = load_config("training", args.config)
    assert isinstance(config, TrainingConfig)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("fitness_contrastive_finetune")

    with mlflow.start_run(run_name=config.run_name or "contrastive-finetune"):
        mlflow.log_param("model_name", config.model_name)
        mlflow.log_param("loss_function", config.loss_function.value)
        mlflow.log_param("num_epochs", config.num_epochs)
        mlflow.log_param("batch_size", config.data.batch_size)
        mlflow.log_param("learning_rate", config.optimizer.learning_rate)

        finetuned_dir = run_training(config)

        results = compare_before_after(
            config, finetuned_dir, args.dataset, args.embeddings_dir, args.testset, args.top_k
        )
        for system, metrics in results.items():
            for metric_name, value in metrics.items():
                if isinstance(value, int | float):
                    raw_name = f"{system}_{metric_name}".replace("@", "_at_")
                    # mlflow metric names allow only alnum/_/-/./space/:// — strip
                    # everything else (e.g. the "(off-the-shelf)" parens) rather
                    # than fail the whole run over a cosmetic label.
                    safe_name = re.sub(r"[^A-Za-z0-9_\-. :/]", "", raw_name)
                    mlflow.log_metric(safe_name, value)

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
