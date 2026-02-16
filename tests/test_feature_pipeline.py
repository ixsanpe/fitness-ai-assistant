"""Tests for feature pipeline components."""

import json
import tempfile
from pathlib import Path

import pytest

from src.config import EmbeddingConfig, FeatureConfig
from src.config.schemas import EmbeddingType
from src.feature_pipeline.embedders import create_embedder
from src.feature_pipeline.loaders import ExerciseLoader


class TestFeaturePipeline:
    """Test feature pipeline components."""

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
                "force": "push",
                "level": "beginner",
                "mechanic": "compound",
                "equipment": "bodyweight",
                "primaryMuscles": ["chest", "triceps"],
                "secondaryMuscles": ["shoulders"],
                "category": "strength",
                "instructions": ["Get into plank position", "Lower body", "Push up"],
            }

            (exercise_dir / "metadata.json").write_text(json.dumps(metadata))

            yield raw_dir

    def test_exercise_loader_with_config(self, temp_raw_dir):
        """Test ExerciseLoader can load exercises from directory."""
        loader = ExerciseLoader(min_text_length=5, remove_duplicates=False)
        items = loader.load(temp_raw_dir)

        # Should load at least one exercise
        assert len(items) >= 1
        assert "combined_text" in items[0]
        assert "image_paths" in items[0]


class TestEmbedders:
    """Test embedder functionality."""

    @pytest.fixture
    def temp_dataset(self):
        """Create temporary dataset file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
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

    def test_sentence_embedder(self, temp_dataset):
        """Test sentence embedder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config.schemas import PathConfig

            config = FeatureConfig(
                name="test-sentence-embeddings",
                embedding=EmbeddingConfig(embedding_type=EmbeddingType.SENTENCE),
                paths=PathConfig(embeddings_dir=Path(tmpdir)),
            )

            # Load dataset items from JSONL
            items = []
            with open(temp_dataset) as f:
                for line in f:
                    items.append(json.loads(line))

            embedder = create_embedder(config, device="cpu")

            # Should not raise
            embeddings = embedder.compute(items)

            # Check output shape
            assert embeddings.shape[0] == 2  # Two samples
            assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension

    def test_clip_embedder(self, temp_dataset):
        """Test CLIP embedder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config.schemas import PathConfig

            config = FeatureConfig(
                name="test-clip-embeddings",
                embedding=EmbeddingConfig(
                    embedding_type=EmbeddingType.CLIP, model_name="openai/clip-vit-base-patch32"
                ),
                paths=PathConfig(embeddings_dir=Path(tmpdir)),
            )

            # Load dataset items from JSONL
            items = []
            with open(temp_dataset) as f:
                for line in f:
                    items.append(json.loads(line))

            embedder = create_embedder(config, device="cpu")

            # Should not raise
            embeddings = embedder.compute(items)

            # Check output shape
            assert embeddings.shape[0] == 2  # Two samples
            assert embeddings.shape[1] == 1024  # CLIP dimension


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
