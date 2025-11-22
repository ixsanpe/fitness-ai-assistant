from pymilvus import MilvusClient
import torch
import numpy as np
from pathlib import Path
from src.data.compute_embeddings import load_dataset

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