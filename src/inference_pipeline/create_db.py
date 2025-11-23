import argparse
import sys
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import InferenceConfig, load_config
from src.feature_pipeline.compute_embeddings import load_dataset, load_embeddings


def create_vector_db(config: InferenceConfig):
    """Create and populate vector database with embeddings.

    Args:
        config: Inference configuration
    """
    print(f"Loading embeddings from: {config.paths.embeddings_dir}")

    # Load embeddings based on model type
    embedding_type = config.model.embedding_type
    if "clip" in config.model.model_name.lower():
        embedding_type = "clip"
    else:
        embedding_type = "sentence"

    embeddings = load_embeddings(config.paths.embeddings_dir, embedding_type)

    if embeddings is None:
        print(f"Error: No embeddings found for type '{embedding_type}'")
        print("Please run compute_embeddings.py first")
        return

    vectors = embeddings.cpu().numpy().astype(np.float32)
    print(f"Loaded embeddings: {vectors.shape}")
    dim = vectors.shape[1]

    # Initialize Milvus client
    if config.vector_db.backend == "milvus_lite":
        db_path = config.vector_db.local_path or "milvus_demo.db"
    else:
        raise ValueError(f"Unsupported vector DB backend: {config.vector_db.backend}")
    print(f"Connecting to Milvus: {db_path}")
    client = MilvusClient(str(db_path))

    # Drop existing collection if it exists
    collection_name = config.vector_db.collection_name
    if client.has_collection(collection_name=collection_name):
        print(f"Dropping existing collection: {collection_name}")
        client.drop_collection(collection_name=collection_name)

    # Create new collection
    print(f"Creating collection: {collection_name} (dimension: {dim})")
    client.create_collection(
        collection_name=collection_name,
        dimension=dim,
    )

    # Load dataset
    dataset_path = config.dataset_path
    print(f"Loading dataset from: {dataset_path}")
    items = load_dataset(dataset_path)

    if len(items) != len(vectors):
        print(f"Warning: Dataset size ({len(items)}) != embeddings size ({len(vectors)})")
        min_len = min(len(items), len(vectors))
        items = items[:min_len]
        vectors = vectors[:min_len]
        print(f"Using first {min_len} items")

    # Prepare data for insertion
    print(f"Preparing {len(vectors)} items for insertion...")
    data = [
        {
            "id": i,
            "vector": vectors[i],
            "text": items[i]["combined_text"],
            "subject": "fitness",
            "name": items[i].get("id", f"exercise_{i}"),
        }
        for i in range(len(vectors))
    ]

    # Insert data
    print("Inserting data into Milvus...")
    res = client.insert(collection_name=collection_name, data=data)

    print(
        f"✅ Successfully inserted {res['insert_count']} items into collection '{collection_name}'"
    )
    print(f"Database location: {db_path}")


def main(args):
    """Main function to create vector database."""
    # Load configuration
    config: InferenceConfig = load_config("inference", args.config)

    # Override collection name if provided
    if args.collection_name:
        config.vector_db.collection_name = args.collection_name

    print(f"Configuration: {config.name}")
    print(f"Vector DB Backend: {config.vector_db.backend}")
    print(f"Collection Name: {config.vector_db.collection_name}")

    create_vector_db(config)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create and populate vector database")
    p.add_argument("--config", type=str, default=None, help="Path to config file")
    p.add_argument("--collection_name", type=str, default=None, help="Override collection name")
    args = p.parse_args()
    main(args)

# Usage examples:
# python src/inference_pipeline/create_db.py --config configs/inference.yaml
# python src/inference_pipeline/create_db.py --config configs/inference_prod.yaml
# python src/inference_pipeline/create_db.py --config configs/inference.yaml --collection_name my_exercises
