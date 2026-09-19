"""Create and populate vector database with embeddings.

Usage:
    python -m src.inference_pipeline.create_db --config configs/inference.yaml
"""

import argparse

import numpy as np
from pymilvus import MilvusClient

from src.config import InferenceConfig, load_config
from src.inference_pipeline.loaders import load_embeddings, load_metadata


def create_vector_db(config):
    """Create and populate vector database with embeddings.

    Args:
        config: InferenceConfig instance
    """
    print(f"📦 Creating vector database: {config.vector_db.collection_name}")
    print(f"Loading embeddings from: {config.paths.embeddings_dir}")

    # Load embeddings
    embedding_type = config.model.embedding_type
    embeddings = load_embeddings(config.paths.embeddings_dir, embedding_type)
    vectors = embeddings.astype(np.float32)
    dim = vectors.shape[1]

    # Initialize Milvus client
    if config.vector_db.backend == "milvus_lite":
        db_path = config.vector_db.local_path or "milvus_demo.db"
    else:
        raise ValueError(f"Unsupported backend: {config.vector_db.backend}")

    print(f"Connecting to Milvus: {db_path}")
    client = MilvusClient(str(db_path))

    # Drop existing collection if it exists
    collection_name = config.vector_db.collection_name
    if client.has_collection(collection_name=collection_name):
        print(f"Dropping existing collection: {collection_name}")
        client.drop_collection(collection_name=collection_name)

    # Create new collection
    print(f"Creating collection: {collection_name} (dimension: {dim})")
    client.create_collection(collection_name=collection_name, dimension=dim)

    # Load metadata
    dataset_path = config.dataset_path
    print(f"Loading dataset from: {dataset_path}")
    items = load_metadata(dataset_path)

    if len(items) != len(vectors):
        print(f"⚠️  Dataset size ({len(items)}) != embeddings size ({len(vectors)})")
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
            "text": items[i].get("combined_text", ""),
            "subject": "fitness",
            "name": items[i].get("id", f"exercise_{i}"),
        }
        for i in range(len(vectors))
    ]

    # Insert data
    print("Inserting data into Milvus...")
    res = client.insert(collection_name=collection_name, data=data)

    print(f"✅ Successfully inserted {res['insert_count']} items into '{collection_name}'")
    print(f"Database location: {db_path}")


def main(args):
    """Main function."""
    config = load_config("inference", args.config)
    assert isinstance(config, InferenceConfig), "Config must be an InferenceConfig instance"

    print(f"📋 Configuration: {config.name}")
    print(f"Backend: {config.vector_db.backend}")
    print(f"Collection: {config.vector_db.collection_name}\n")

    create_vector_db(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create vector database from embeddings")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Config file")
    args = parser.parse_args()
    main(args)
