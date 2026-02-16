"""Feature pipeline orchestrator."""

from pathlib import Path

import torch

from src.config import FeatureConfig, load_config
from src.feature_pipeline.embedders import create_embedder
from src.feature_pipeline.loaders import ExerciseLoader
from src.feature_pipeline.storage import EmbeddingStorage


class FeaturePipeline:
    """Orchestrate the full feature pipeline: load data, compute embeddings."""

    def __init__(self, config: FeatureConfig, device: str | None = None):
        self.config = config
        self.device = device or self._get_device()
        self.loader = ExerciseLoader(
            min_text_length=config.dataset.min_text_length,
            remove_duplicates=config.dataset.remove_duplicates,
        )
        self.embedder = create_embedder(config, self.device)
        self.storage = EmbeddingStorage(config.paths.embeddings_dir)

    def _get_device(self) -> str:
        """Determine device to use."""
        if self.config.device.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.config.device.device

    def run(
        self,
        input_dir: Path | str | None = None,
        dataset_path: Path | str | None = None,
        sample_limit: int | None = None,
    ) -> dict:
        """Run the full pipeline.

        Args:
            input_dir: Override input directory for raw data
            dataset_path: Override dataset path (if already built)
            sample_limit: Limit number of items to process

        Returns:
            Dictionary with pipeline results
        """
        self.config.ensure_directories()

        # Step 1: Load or build dataset
        if dataset_path:
            print(f"Loading existing dataset: {dataset_path}")
            items = self._load_jsonl(dataset_path)
        else:
            input_path = Path(input_dir or self.config.dataset.input_path)
            print(f"Building dataset from: {input_path}")
            self.loader.sample_limit = sample_limit
            items = self.loader.load(input_path)
            self.loader.save(items, self.config.dataset.output_path)

        # Step 2: Compute embeddings
        print(f"\nComputing embeddings using device: {self.device}")
        embeddings = self.embedder.compute(items)

        return {
            "items": items,
            "embeddings": embeddings,
            "device": self.device,
            "num_items": len(items),
            "embedding_shape": embeddings.shape,
        }

    def _load_jsonl(self, path: Path | str) -> list[dict]:
        """Load JSONL dataset."""
        import json

        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        return items


def run_pipeline(config_path: str | None = None, **kwargs) -> dict:
    """Convenience function to run the pipeline.

    Args:
        config_path: Path to config file
        **kwargs: Additional arguments for pipeline.run()

    Returns:
        Pipeline results
    """
    config = load_config("feature", config_path or "config.yaml")
    assert isinstance(config, FeatureConfig)
    pipeline = FeaturePipeline(config)
    return pipeline.run(**kwargs)
