"""
MLflow experiments for comparing embedding models and configurations.

This module provides functionality to run experiments comparing:
- Different embedding models (sentence-transformers vs CLIP)
- Different model architectures within each type
- Different hyperparameters (batch size, normalization, etc.)
- Different text processing strategies

The experiments track metrics like:
- Embedding dimension
- Inference time
- Memory usage
- Embedding quality metrics (if applicable)
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import torch

from src.config import FeatureConfig, load_config
from src.feature_pipeline.embedders import create_embedder
from src.feature_pipeline.utils import load_dataset


class EmbeddingExperiment:
    """Base class for embedding experiments with MLflow tracking."""

    def __init__(self, experiment_name: str, tracking_uri: str | None = None):
        """Initialize experiment tracking.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI
                - Local: "file:./mlruns" (default)
                - Remote server: "http://localhost:5000" or "https://your-server.com"
                - Database: "sqlite:///mlflow.db" or "postgresql://user:pass@host/db"
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("file:./mlruns")

        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name

    def run_experiment(
        self,
        config: FeatureConfig,
        items: list[dict],
        max_samples: int | None = None,
    ):
        """Run experiment using the feature pipeline embedder.

        Args:
            config: Feature pipeline configuration
            items: Dataset items
            device: Device to use (cpu, cuda, mps)
            max_samples: Maximum number of samples to use (for faster testing)
        """
        sample_items = items[:max_samples] if max_samples else items
        run_name = (
            f"{config.embedding.embedding_type}_"
            f"{config.embedding.model_name.replace('/', '_')}_"
            f"bs{config.embedding.batch_size}"
        )
        device = config.device.resolve()

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("embedding_type", config.embedding.embedding_type)
            mlflow.log_param("model_name", config.embedding.model_name)
            mlflow.log_param("batch_size", config.embedding.batch_size)
            mlflow.log_param("normalize", config.embedding.normalize)
            mlflow.log_param("device", device)
            mlflow.log_param("num_samples", len(sample_items))

            try:
                print(f"\n{'=' * 60}")
                print(f"Running experiment: {run_name}")
                print(f"{'=' * 60}")
                print(device)

                embedder = create_embedder(config, device)

                encode_start = time.time()
                embeddings = embedder.compute(sample_items)
                encode_time = time.time() - encode_start

                mlflow.log_metric("encoding_time_seconds", encode_time)
                mlflow.log_metric("samples_per_second", len(sample_items) / encode_time)
                mlflow.log_metric("embedding_dimension", embeddings.shape[-1])
                mlflow.log_metric("embedding_mean", float(embeddings.mean()))
                mlflow.log_metric("embedding_std", float(embeddings.std()))
                mlflow.log_metric("embedding_min", float(embeddings.min()))
                mlflow.log_metric("embedding_max", float(embeddings.max()))

                if device == "cuda" and torch.cuda.is_available():
                    memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
                    mlflow.log_metric("max_memory_mb", memory_allocated)
                    torch.cuda.reset_peak_memory_stats()
                elif device == "mps" and torch.backends.mps.is_available():
                    memory_allocated = torch.mps.current_allocated_memory() / 1024**2  # MB
                    mlflow.log_metric("max_memory_mb", memory_allocated)

                self._log_embedding_quality_metrics(embeddings)

                print(f"✅ Experiment completed: {run_name}")
                print(f"   - Encoding time: {encode_time:.2f}s")
                print(f"   - Samples/sec: {len(sample_items) / encode_time:.2f}")
                print(f"   - Embedding dim: {embeddings.shape[-1]}")

                return embeddings

            except Exception as e:
                print(f"Error in experiment {run_name}: {e}")
                mlflow.log_param("error", str(e))
                raise

    def _log_embedding_quality_metrics(self, embeddings: torch.Tensor):
        """Log embedding quality metrics.

        Args:
            embeddings: Tensor of embeddings
        """
        # Convert to numpy for calculations
        emb_np = embeddings.cpu().numpy()

        # Compute pairwise cosine similarities (on a sample for efficiency)
        sample_size = min(100, len(emb_np))
        sample_emb = emb_np[:sample_size]

        # Normalize for cosine similarity
        sample_norm = sample_emb / (np.linalg.norm(sample_emb, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(sample_norm, sample_norm.T)

        # Remove diagonal (self-similarities)
        mask = np.ones_like(similarities, dtype=bool)
        np.fill_diagonal(mask, False)
        off_diagonal_sims = similarities[mask]

        mlflow.log_metric("avg_cosine_similarity", float(off_diagonal_sims.mean()))
        mlflow.log_metric("std_cosine_similarity", float(off_diagonal_sims.std()))
        mlflow.log_metric("min_cosine_similarity", float(off_diagonal_sims.min()))
        mlflow.log_metric("max_cosine_similarity", float(off_diagonal_sims.max()))

        # Log embedding norm statistics
        norms = np.linalg.norm(emb_np, axis=1)
        mlflow.log_metric("avg_embedding_norm", float(norms.mean()))
        mlflow.log_metric("std_embedding_norm", float(norms.std()))

    def run_comparison_suite(
        self,
        config_paths: list[str],
        dataset_path: Path,
        device: str = "cpu",
        max_samples: int | None = None,
    ):
        """Run a suite of experiments comparing different configurations.

        Args:
            config_paths: List of config file paths to compare
            dataset_path: Path to the dataset
            device: Device to use
            max_samples: Maximum samples for faster testing
        """
        items = load_dataset(dataset_path)

        print(f"\n{'#' * 60}")
        print(f"Starting comparison suite with {len(config_paths)} configurations")
        print(f"Dataset: {len(items)} items (using {max_samples or 'all'})")
        print(f"Device: {device}")
        print(f"{'#' * 60}\n")

        for config_path in config_paths:
            try:
                config = load_config("feature", config_path)
                assert isinstance(config, FeatureConfig), (  # noqa
                    "Config file must be of type FeatureConfig"
                )
                self.run_experiment(config=config, items=items, max_samples=max_samples)
            except Exception as e:
                print(f"Failed to run experiment with config {config_path}: {e}")
                continue

        print(f"\n{'#' * 60}")
        print("Comparison suite completed!")
        print("View results with: mlflow ui")
        print(f"{'#' * 60}\n")


def main(args):
    """Main function to run MLflow experiments."""
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Create experiment tracker
    experiment = EmbeddingExperiment(
        experiment_name=args.experiment_name, tracking_uri=args.tracking_uri
    )

    # Load dataset
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if args.mode == "single":
        # Run single experiment from config
        items = load_dataset(dataset_path)
        config = load_config("feature", args.config)
        assert isinstance(config, FeatureConfig), "Config file must be of type FeatureConfig"
        experiment.run_experiment(config=config, items=items, max_samples=args.max_samples)

    elif args.mode == "compare":
        # Run comparison suite with multiple configs
        experiment.run_comparison_suite(
            config_paths=args.configs,
            dataset_path=dataset_path,
            device=device,
            max_samples=args.max_samples,
        )

    print("\n✅ All experiments completed!")
    print("\nTo view results, run:")
    print("  mlflow ui")
    print("\nThen open: http://localhost:5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MLflow experiments for embedding model comparison"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "compare"],
        default="compare",
        help="Run single experiment or comparison suite",
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="fitness_embeddings",
        help="MLflow experiment name",
    )

    parser.add_argument(
        "--tracking_uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="MLflow tracking URI. Examples: 'http://localhost:5000', 'file:./mlruns', 'sqlite:///mlflow.db'",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/feature_sentence.yaml",
        help="Config file for single mode",
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/feature_sentence.yaml", "configs/feature_sentence_2.yaml"],
        help="List of config files for compare mode",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/processed/exercises_dataset.jsonl",
        help="Path to dataset",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "auto", None],
        default=None,
        help="Device to use for computation",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for faster testing)",
    )

    args = parser.parse_args()
    main(args)

# Usage examples:
#
# Run single experiment (local):
# python -m src.training_pipeline.mlflow_experiments --mode single --config configs/feature_sentence.yaml
#
# Compare multiple configurations (local):
# python -m src.training_pipeline.mlflow_experiments --mode compare --configs configs/feature_sentence.yaml configs/feature_clip.yaml
#
# Quick test with limited samples:
# python -m src.training_pipeline.mlflow_experiments --mode compare --max_samples 100
#
# Use remote MLflow server:
# python -m src.training_pipeline.mlflow_experiments --mode compare --tracking_uri http://localhost:5000
#
# Use with database backend:
# python -m src.training_pipeline.mlflow_experiments --mode compare --tracking_uri sqlite:///mlflow.db
#
# Start MLflow server (in separate terminal):
# mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
#
# View local results:
# mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db
