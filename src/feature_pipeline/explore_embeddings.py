
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import argparse

from src.data.constants import EMBEDDINGS_OUT_DIR
from src.data.compute_embeddings import load_dataset, load_embeddings
EMBEDDINGS_OUT_PATH = Path(EMBEDDINGS_OUT_DIR)

def find_similar(query_idx, embeddings, exercises, top_k=10, plot_tsne=False):
    query_emb = embeddings[query_idx].unsqueeze(0)  # [1, 384]
    
    # Cosine similarity in PyTorch
    similarities = torch.nn.functional.cosine_similarity(
        query_emb, embeddings, dim=1
    )
    
    top_indices = torch.argsort(similarities, descending=True)[:top_k]
    top_scores = similarities[top_indices]
    
    similar_idx = top_indices.cpu().numpy()
    scores = top_scores.cpu().numpy()

    if plot_tsne:
        print(f"\nExercises similar to {exercises[query_idx]['id']}:")
        for idx, score in zip(similar_idx, scores):
            print(f"  {exercises[idx]['id']}: {score:.3f}, {exercises[idx]['attributes']['force']}, {exercises[idx]['attributes']['category']}, {exercises[idx]['attributes']['primaryMuscles']}")
    
        # Move to CPU for visualization
        embeddings_np = embeddings.cpu().numpy()

        # Visualize with t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        coords_2d = tsne.fit_transform(embeddings_np)

        plt.figure(figsize=(12, 8))
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.6)
        # Visualize the top_indices in a different color
        plt.scatter(coords_2d[similar_idx, 0], coords_2d[similar_idx, 1], color='red', alpha=0.6)
        plt.title("Exercise Embedding Space (PyTorch)")
        plt.show()

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    items = load_dataset(args.dataset_path)

    embeddings = load_embeddings(args.embed_tool)
    print(args.embed_tool, " Embeddings computed:", embeddings.shape)

    find_similar(0, embeddings, items, top_k=10, plot_tsne=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--embed_tool", default="sentence")  # or "clip"
    args = p.parse_args()
    main(args)

# python -m src.data.explore_embeddings --dataset_path data/processed/exercises_dataset.jsonl --embed_tool "clip"