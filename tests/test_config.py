"""Tests for configuration system."""

import pytest

from src.config import (
    FeatureConfig,
    InferenceConfig,
    TrainingConfig,
    load_config,
)


class TestConfigValidation:
    """Test that config classes validate correctly."""

    def test_feature_config_defaults(self):
        """Test FeatureConfig can be created with defaults."""
        config = FeatureConfig(name="test-feature")
        assert config.name == "test-feature"
        assert config.embedding.model_name is not None
        assert config.paths.project_root is not None

    def test_inference_config_defaults(self):
        """Test InferenceConfig can be created with defaults."""
        config = InferenceConfig(name="test-inference")
        assert config.name == "test-inference"
        assert config.model.model_name is not None
        assert config.vector_db.backend is not None

    def test_training_config_defaults(self):
        """Test TrainingConfig can be created with defaults."""
        config = TrainingConfig(name="test-training")
        assert config.name == "test-training"
        assert config.optimizer.learning_rate > 0


class TestConfigYAMLMatching:
    """Test that YAML configs match Pydantic models."""

    def test_feature_sentence_yaml_loads(self):
        """Test feature_sentence.yaml loads correctly."""
        config = load_config("feature", "configs/feature_sentence.yaml")
        assert config.embedding.embedding_type == "sentence"
        assert config.embedding.model_name == "all-MiniLM-L6-v2"

    def test_feature_clip_yaml_loads(self):
        """Test feature_clip.yaml loads correctly."""
        config = load_config("feature", "configs/feature_clip.yaml")
        assert config.embedding.embedding_type == "clip"
        assert config.embedding.model_name == "openai/clip-vit-base-patch32"

    def test_inference_yaml_loads(self):
        """Test inference.yaml loads correctly."""
        config = load_config("inference", "configs/inference.yaml")
        assert config.vector_db.backend == "milvus_lite"


class TestConfigPaths:
    """Test that paths in configs are valid."""

    def test_feature_paths_structure(self):
        """Test feature config has required path fields."""
        config = load_config("feature", "configs/feature_sentence.yaml")
        assert hasattr(config.paths, "project_root")
        assert hasattr(config.paths, "raw_data_dir")
        assert hasattr(config.paths, "processed_data_dir")
        assert hasattr(config.paths, "embeddings_dir")
        assert hasattr(config.dataset, "output_path")

    def test_inference_paths_structure(self):
        """Test inference config has required path fields."""
        config = load_config("inference", "configs/inference.yaml")
        assert hasattr(config.paths, "project_root")
        assert hasattr(config, "dataset_path")
        assert hasattr(config.paths, "embeddings_dir")


class TestConfigEnums:
    """Test that enum values are validated."""

    def test_invalid_embedding_type(self):
        """Test that invalid embedding type raises error."""
        from src.config.feature_config import EmbeddingType

        with pytest.raises(ValueError):
            EmbeddingType("invalid_type")

    def test_valid_embedding_types(self):
        """Test all valid embedding types."""
        from src.config.feature_config import EmbeddingType

        assert EmbeddingType.SENTENCE == "sentence"
        assert EmbeddingType.CLIP == "clip"
        assert EmbeddingType.CLIP_TEXT == "clip_text"
        assert EmbeddingType.CLIP_IMAGE == "clip_image"

    def test_valid_backend_types(self):
        """Test all valid backend types."""
        from src.config.inference_config import VectorDBBackend

        assert VectorDBBackend.LOCAL == "local"
        assert VectorDBBackend.MILVUS == "milvus"


class TestConfigValidators:
    """Test Pydantic validators work correctly."""

    def test_positive_batch_size(self):
        """Test batch size must be positive."""
        config = FeatureConfig(name="test")
        config.embedding.batch_size = 32
        assert config.embedding.batch_size == 32

        # Pydantic should validate on creation
        with pytest.raises(ValueError):
            FeatureConfig(name="test", embedding={"batch_size": -1})

    def test_positive_learning_rate(self):
        """Test learning rate must be positive."""
        config = TrainingConfig(name="test")
        assert config.optimizer.learning_rate > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
