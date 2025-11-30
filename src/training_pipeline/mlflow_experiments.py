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
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FeatureConfig, load_config
from src.feature_pipeline.compute_embeddings import load_dataset


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

    def run_sentence_experiment(
        self,
        model_name: str,
        items: list[dict],
        batch_size: int = 32,
        normalize: bool = True,
        device: str = "cpu",
        max_samples: int | None = None,
    ):
        """Run experiment for sentence-transformer model.

        Args:
            model_name: HuggingFace model name
            items: Dataset items
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings
            device: Device to use (cpu, cuda)
            max_samples: Maximum number of samples to use (for faster testing)
        """
        run_name = f"sentence_{model_name.replace('/', '_')}_bs{batch_size}"

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_param("embedding_type", "sentence")
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("normalize", normalize)
            mlflow.log_param("device", device)
            mlflow.log_param("num_samples", len(items[:max_samples] if max_samples else items))

            try:
                # Load model
                print(f"\n{'='*60}")
                print(f"Running experiment: {run_name}")
                print(f"{'='*60}")
                # TODO: reuse the EmbeddingGenerator class from feature_pipeline.compute_embeddings
                load_start = time.time()
                model = SentenceTransformer(model_name, device=device)
                load_time = time.time() - load_start

                mlflow.log_metric("model_load_time_seconds", load_time)
                mlflow.log_metric("embedding_dimension", model.get_sentence_embedding_dimension())

                # Prepare texts
                texts = [
                    it.get("combined_text", "")
                    for it in (items[:max_samples] if max_samples else items)
                ]

                # Measure encoding time
                encode_start = time.time()

                embeddings = model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    normalize_embeddings=normalize,
                )
                encode_time = time.time() - encode_start

                # Log metrics
                mlflow.log_metric("encoding_time_seconds", encode_time)
                mlflow.log_metric("samples_per_second", len(texts) / encode_time)
                mlflow.log_metric("embedding_mean", float(embeddings.mean()))
                mlflow.log_metric("embedding_std", float(embeddings.std()))
                mlflow.log_metric("embedding_min", float(embeddings.min()))
                mlflow.log_metric("embedding_max", float(embeddings.max()))

                # Memory metrics (if using CUDA)
                if device == "cuda" and torch.cuda.is_available():
                    memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
                    mlflow.log_metric("max_memory_mb", memory_allocated)
                    torch.cuda.reset_peak_memory_stats()

                # Compute embedding quality metrics
                self._log_embedding_quality_metrics(embeddings)

                # Infer model signature for MLflow
                sample_texts = texts[:3]  # Use small sample for signature
                sample_output = model.encode(sample_texts, convert_to_numpy=True)
                signature = mlflow.models.infer_signature(
                    model_input=sample_texts,
                    model_output=sample_output,
                )
                print(signature)

                # Log model with sentence-transformers flavor (MLflow built-in support)
                mlflow.sentence_transformers.log_model(
                    model,
                    name=model_name,
                    signature=signature,
                    input_example=sample_texts,
                    registered_model_name=None,  # Set to register the model
                )

                print(f"✅ Experiment completed: {run_name}")
                print(f"   - Encoding time: {encode_time:.2f}s")
                print(f"   - Samples/sec: {len(texts) / encode_time:.2f}")
                print(f"   - Embedding dim: {model.get_sentence_embedding_dimension()}")

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

        print(f"\n{'#'*60}")
        print(f"Starting comparison suite with {len(config_paths)} configurations")
        print(f"Dataset: {len(items)} items (using {max_samples or 'all'})")
        print(f"Device: {device}")
        print(f"{'#'*60}\n")

        for config_path in config_paths:
            try:
                config: FeatureConfig = load_config("feature", config_path)

                if config.embedding.embedding_type == "sentence":
                    self.run_sentence_experiment(
                        model_name=config.embedding.model_name,
                        items=items,
                        batch_size=config.embedding.batch_size,
                        normalize=config.embedding.normalize,
                        device=device,
                        max_samples=max_samples,
                    )
                elif config.embedding.embedding_type == "clip":
                    raise NotImplementedError("CLIP experiment not implemented yet")

            except Exception as e:
                print(f"Failed to run experiment with config {config_path}: {e}")
                continue

        print(f"\n{'#'*60}")
        print("Comparison suite completed!")
        print("View results with: mlflow ui")
        print(f"{'#'*60}\n")


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
        config: FeatureConfig = load_config("feature", args.config)

        if config.embedding.embedding_type == "sentence":
            experiment.run_sentence_experiment(
                model_name=config.embedding.model_name,
                items=items,
                batch_size=config.embedding.batch_size,
                normalize=config.embedding.normalize,
                device=device,
                max_samples=args.max_samples,
            )
        elif config.embedding.embedding_type == "clip":
            raise NotImplementedError("CLIP experiment not implemented yet")

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
        default="sqlite:///mlruns/mlflow.db",
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
        choices=["cpu", "cuda", "mps", "auto"],
        default="auto",
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
# python src/training_pipeline/mlflow_experiments.py --mode single --config configs/feature_sentence.yaml
#
# Compare multiple configurations (local):
# python src/training_pipeline/mlflow_experiments.py --mode compare --configs configs/feature_sentence.yaml configs/feature_clip.yaml
#
# Quick test with limited samples:
# python src/training_pipeline/mlflow_experiments.py --mode compare --max_samples 100
#
# Use remote MLflow server:
# python src/training_pipeline/mlflow_experiments.py --mode compare --tracking_uri http://localhost:5000
#
# Use with database backend:
# python src/training_pipeline/mlflow_experiments.py --mode compare --tracking_uri sqlite:///mlflow.db
#
# Start MLflow server (in separate terminal):
# mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
#
# View local results:
# mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db
