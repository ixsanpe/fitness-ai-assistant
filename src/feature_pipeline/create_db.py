from pymilvus import MilvusClient
import torch
import numpy as np
from pathlib import Path
from src.data.compute_embeddings import load_dataset

EMBEDDINGS_OUT_DIR = Path("data/processed/embeddings")

#existing = list(EMBEDDINGS_OUT_DIR.glob("*clip.npy"))
existing = list(EMBEDDINGS_OUT_DIR.glob("*text.npy"))
if existing:
    text_path = existing[0]
    print(f"Found embeddings file: {text_path}")
    vectors = np.load(str(text_path)).astype(np.float32)
print("Dim:", vectors.shape)  # Dim: (873, 1024) or (873, 384)
dim = vectors.shape[1]

client = MilvusClient("milvus_demo.db")
if client.has_collection(collection_name="exercises_embeddings"):
    client.drop_collection(collection_name="exercises_embeddings")
client.create_collection(
    collection_name="exercises_embeddings",
    dimension=dim,  # The vectors we will use in this demo has 768 dimensions
)
dataset_path = "data/processed/exercises_dataset.jsonl"
items = load_dataset(dataset_path)
data = [
    {"id": i, "vector": vectors[i], "text": items[i]["combined_text"], "subject": "fitness"}
    for i in range(len(vectors))
]
res = client.insert(collection_name="exercises_embeddings", data=data)

print(res)