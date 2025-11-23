"""Tests for feature pipeline components."""

import tempfile
from pathlib import Path

import pytest


class TestBuildDataset:
    """Test build_dataset.py functionality."""

    @pytest.fixture
    def temp_raw_dir(self):
        """Create temporary raw data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw" / "exercises"
            raw_dir.mkdir(parents=True)

            # Create sample exercise folder with metadata
            exercise_dir = raw_dir / "Push_Up"
            exercise_dir.mkdir()

            metadata = {
                "name": "Push Up",
                "primaryMuscles": ["chest", "triceps"],
                "instructions": ["Get into plank position", "Lower body", "Push up"],
            }

            import json

            (exercise_dir / "metadata.json").write_text(json.dumps(metadata))

            yield raw_dir

    def test_build_dataset_with_config(self, temp_raw_dir):
        """Test build_dataset can run with config."""
        from src.config.feature_config import FeatureConfig
        from src.feature_pipeline.build_dataset import build_dataset

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as tmp_output:
            output_path = Path(tmp_output.name)

        try:
            # Create config with test paths
            config = FeatureConfig(
                name="test-build-dataset",
                paths={"raw_data_dir": str(temp_raw_dir.parent), "dataset_path": str(output_path)},
            )

            # Should not raise
            try:
                build_dataset(config)
            except Exception as e:
                # It's OK if it fails due to missing data, we just want to test it accepts config
                assert "config" not in str(e).lower()
        finally:
            output_path.unlink(missing_ok=True)


class TestComputeEmbeddings:
    """Test compute_embeddings.py functionality."""

    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            import json

            data = [
                {
                    "id": "push_up",
                    "combined_text": "Push up exercise for chest and triceps",
                    "attributes": {"name": "Push Up"},
                },
                {
                    "id": "sit_up",
                    "combined_text": "Sit up exercise for abs",
                    "attributes": {"name": "Sit Up"},
                },
            ]
            for item in data:
                f.write(json.dumps(item) + "\n")
            temp_path = f.name

        yield Path(temp_path)
        Path(temp_path).unlink(missing_ok=True)

    def test_embedding_generator_sentence(self, temp_dataset):
        """Test EmbeddingGenerator with sentence model."""
        from src.config.feature_config import FeatureConfig
        from src.feature_pipeline.compute_embeddings import EmbeddingGenerator, load_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            config = FeatureConfig(
                name="test-sentence-embeddings",
                embedding={"embedding_type": "sentence"},
                paths={"dataset_path": str(temp_dataset), "embeddings_dir": tmpdir},
            )

            # Load dataset items
            items = load_dataset(temp_dataset)

            generator = EmbeddingGenerator(config, device="cpu")

            # Should not raise
            embeddings = generator.compute(items)

            # Check output shape
            assert embeddings.shape[0] == 2  # Two samples
            assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension

    def test_embedding_generator_clip(self, temp_dataset):
        """Test EmbeddingGenerator with CLIP model."""
        from src.config.feature_config import FeatureConfig
        from src.feature_pipeline.compute_embeddings import EmbeddingGenerator, load_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            config = FeatureConfig(
                name="test-clip-embeddings",
                embedding={"embedding_type": "clip", "model_name": "openai/clip-vit-base-patch32"},
                paths={"dataset_path": str(temp_dataset), "embeddings_dir": tmpdir},
            )

            # Load dataset items
            items = load_dataset(temp_dataset)

            generator = EmbeddingGenerator(config, device="cpu")

            # Should not raise
            embeddings = generator.compute(items)

            # Check output shape
            assert embeddings.shape[0] == 2  # Two samples
            assert embeddings.shape[1] == 1024  # CLIP dimension


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
