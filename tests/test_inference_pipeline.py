"""Tests for inference pipeline components."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import load_config


class TestInferencePipeline:
    """Test inference pipeline functionality."""

    def test_pipeline_loads_config(self):
        """Test pipeline can load config."""
        from src.inference_pipeline.pipeline import InferencePipeline

        # Mock Milvus backend initialization to avoid actual DB connection
        with patch("src.inference_pipeline.pipeline.MilvusBackend") as mock_backend:
            mock_instance = MagicMock()
            mock_backend.return_value = mock_instance

            # Should load config without error
            config = load_config("inference", "configs/inference.yaml")
            pipeline = InferencePipeline(config)

            # Verify backend was initialized
            assert mock_backend.called
            pipeline.close()

    def test_pipeline_query_interface(self):
        """Test pipeline query returns expected format."""
        from src.inference_pipeline.pipeline import InferencePipeline

        with patch("src.inference_pipeline.pipeline.LocalBackend") as mock_backend:
            mock_instance = MagicMock()
            mock_instance.search.return_value = [
                {"id": "ex1", "score": 0.95, "combined_text": "Test exercise"}
            ]
            mock_backend.return_value = mock_instance

            config = load_config("inference", "configs/inference.yaml")
            pipeline = InferencePipeline(config)
            results = pipeline.query("test", top_k=5)

            assert isinstance(results, list)
            assert len(results) > 0
            pipeline.close()


class TestConfigIntegration:
    """Integration tests across pipeline components."""

    def test_feature_to_inference_config_compatibility(self):
        """Test feature config output matches inference config input."""
        feature_config = load_config("feature", "configs/feature_sentence.yaml")
        inference_config = load_config("inference", "configs/inference.yaml")

        # Embeddings directory should be same
        assert feature_config.paths.embeddings_dir == inference_config.paths.embeddings_dir

        # Dataset path should be same
        assert feature_config.dataset.output_path == inference_config.dataset_path

    def test_all_configs_have_consistent_paths(self):
        """Test all configs use consistent path structure."""
        configs = [
            load_config("feature", "configs/feature_sentence.yaml"),
            load_config("feature", "configs/feature_clip.yaml"),
            load_config("inference", "configs/inference.yaml"),
        ]

        # All should have project_root
        for config in configs:
            assert hasattr(config.paths, "project_root")
            assert Path(config.paths.project_root).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
