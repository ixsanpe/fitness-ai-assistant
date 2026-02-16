"""Inference pipeline for vector search and recommendation."""

from pathlib import Path

from src.config.inference import InferenceConfig
from src.inference_pipeline.backends import LocalBackend, MilvusBackend
from src.inference_pipeline.loaders import load_embeddings, load_metadata
from src.inference_pipeline.models import TextEmbedder
from src.inference_pipeline.utils import create_vector_db


class InferencePipeline:
    """Inference pipeline: query embedding and semantic search."""

    def __init__(self, config: InferenceConfig):
        """Initialize pipeline from config.

        Args:
            config: InferenceConfig instance with all settings
        """
        self.config = config
        device = config.device.device
        # Convert DeviceType enum to string if needed
        self.device = device.value if hasattr(device, "value") else device
        self.embedding_type = config.model.embedding_type

        # Initialize text embedder
        self.embedder = TextEmbedder(config.model.model_name, device=self.device)

        # Initialize backend
        backend_name = config.vector_db.backend
        print(f"Initializing {backend_name} backend...")

        if backend_name == "local":
            self._init_local_backend(config)
        elif backend_name in ["milvus", "milvus_lite"]:
            self._init_milvus_backend(config)
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

    def _init_local_backend(self, config: InferenceConfig):
        """Initialize local numpy-based search."""
        embeddings_dir = Path(config.paths.embeddings_dir)
        dataset_path = Path(config.dataset_path)

        # Load data
        embeddings = load_embeddings(embeddings_dir, self.embedding_type)
        metadata = load_metadata(dataset_path)

        # Create backend
        self.backend = LocalBackend(embeddings, metadata)
        print(f"✅ Local backend ready: {len(metadata)} items")

    def _init_milvus_backend(self, config: InferenceConfig):
        """Initialize Milvus-based search."""
        db_path = config.vector_db.local_path or "milvus_demo.db"
        max_retries = 2
        for attempt in range(max_retries):
            try:
                self.backend = MilvusBackend(
                    db_path=db_path,
                    collection_name=config.vector_db.collection_name,
                    vector_field=config.vector_db.vector_field,
                    metric_type=config.vector_db.metric_type.upper(),
                    output_fields=config.vector_db.output_fields or [],
                )
                print("✅ Milvus backend ready")
                return
            except Exception as e:
                if attempt == 0:
                    print(f"❌ Failed to initialize Milvus backend: {e}")
                    print("Attempting to create vector database...")
                    create_vector_db(config)
                else:
                    print(f"❌ Failed to initialize Milvus backend after retry: {e}")
                    raise

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        """Query for semantically similar exercises.

        Args:
            text: Query text
            top_k: Number of results

        Returns:
            List of result dicts with keys: idx, id, score, combined_text, attributes
        """
        # Embed query
        query_vector = self.embedder.embed(text)

        # Search
        results = self.backend.search(query_vector, top_k=top_k)

        return results

    def close(self):
        """Clean up resources."""
        if hasattr(self, "backend"):
            self.backend.close()
