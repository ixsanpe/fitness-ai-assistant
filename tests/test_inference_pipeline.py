"""Tests for inference pipeline components."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import load_config


class TestDemoQuery:
    """Test demo_query.py functionality."""

    def test_demo_query_loads_config(self):
        """Test demo_query can load config."""
        from src.inference_pipeline.demo_query import demo

        # Mock InferencePipeline to avoid actual initialization
        with patch("src.inference_pipeline.demo_query.InferencePipeline") as mock_pipe:
            mock_instance = MagicMock()
            mock_instance.query.return_value = []
            mock_pipe.return_value = mock_instance

            # Should load config without error
            demo(query="test", top_k=5, config_path="configs/inference.yaml")

            # Verify pipeline was initialized with config
            assert mock_pipe.called

    def test_demo_query_with_prod_config(self):
        """Test demo_query with production config."""
        from src.inference_pipeline.demo_query import demo

        with patch("src.inference_pipeline.demo_query.InferencePipeline") as mock_pipe:
            mock_instance = MagicMock()
            mock_instance.query.return_value = [
                {"id": "ex1", "score": 0.95, "combined_text": "Test exercise"}
            ]
            mock_pipe.return_value = mock_instance

            # Should work with prod config
            demo(query="test", top_k=5, config_path="configs/inference_prod.yaml")

            assert mock_pipe.called


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
            load_config("inference", "configs/inference_prod.yaml"),
        ]

        # All should have project_root
        for config in configs:
            assert hasattr(config.paths, "project_root")
            assert Path(config.paths.project_root).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
