from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class InferenceConfig:
    # embedding: 'sentence' or 'clip'
    embedding_type: str = "sentence"
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"

    # backend: 'local' or 'milvus'
    backend: str = "local"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "exercises_embeddings"
    milvus_vector_field: str = "vector"
    milvus_output_fields: List[str] = None

    # dataset / embeddings paths
    dataset_path: Path = Path("data/processed/exercises_dataset.jsonl")
    embeddings_dir: Path = Path("data/processed/embeddings")

    # UI defaults
    top_k: int = 5


def default_config() -> InferenceConfig:
    return InferenceConfig()
