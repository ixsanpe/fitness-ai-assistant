import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FeatureConfig, load_config
from src.feature_pipeline.compute_embeddings import load_dataset, load_embeddings


def find_similar(query_idx, embeddings, exercises, top_k=10, plot_tsne=False):
    """Find similar exercises based on embedding similarity.

    Args:
        query_idx: Index of query exercise
        embeddings: Tensor of all embeddings
        exercises: List of exercise items
        top_k: Number of similar exercises to find
        plot_tsne: Whether to plot t-SNE visualization
    """
    query_emb = embeddings[query_idx].unsqueeze(0)  # [1, dim]

    # Cosine similarity in PyTorch
    similarities = torch.nn.functional.cosine_similarity(query_emb, embeddings, dim=1)

    top_indices = torch.argsort(similarities, descending=True)[:top_k]
    top_scores = similarities[top_indices]

    similar_idx = top_indices.cpu().numpy()
    scores = top_scores.cpu().numpy()

    print(f"\nExercises similar to {exercises[query_idx]['id']}:")
    for idx, score in zip(similar_idx, scores, strict=False):
        attrs = exercises[idx].get("attributes", {})
        print(
            f"  {exercises[idx]['id']}: {score:.3f}, "
            f"force={attrs.get('force', 'N/A')}, "
            f"category={attrs.get('category', 'N/A')}, "
            f"primaryMuscles={attrs.get('primaryMuscles', 'N/A')}"
        )

    if plot_tsne:
        # Move to CPU for visualization
        embeddings_np = embeddings.cpu().numpy()

        # Visualize with t-SNE
        print("\nComputing t-SNE projection...")
        tsne = TSNE(n_components=2, random_state=42)
        coords_2d = tsne.fit_transform(embeddings_np)

        plt.figure(figsize=(12, 8))
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1], alpha=0.6, label="All exercises")
        plt.scatter(
            coords_2d[similar_idx, 0],
            coords_2d[similar_idx, 1],
            color="red",
            alpha=0.8,
            s=100,
            label="Similar exercises",
        )
        plt.scatter(
            coords_2d[query_idx, 0],
            coords_2d[query_idx, 1],
            color="green",
            alpha=1.0,
            s=200,
            marker="*",
            label="Query",
        )
        plt.title(f"Exercise Embedding Space - Query: {exercises[query_idx]['id']}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main(args):
    """Main function to explore embeddings."""
    # Load configuration
    config: FeatureConfig = load_config("feature", args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Embedding type: {config.embedding.embedding_type}")

    # Load dataset
    dataset_path = args.dataset_path or config.dataset.output_path
    items = load_dataset(dataset_path)

    # Load embeddings
    embeddings_dir = config.paths.embeddings_dir
    embeddings = load_embeddings(embeddings_dir, config.embedding.embedding_type)

    if embeddings is None:
        print(f"Error: No embeddings found in {embeddings_dir}")
        print("Please run compute_embeddings.py first")
        return

    print(f"Embeddings loaded: {embeddings.shape}")

    # Find similar exercises
    query_idx = args.query_idx
    if query_idx >= len(items):
        print(f"Error: query_idx {query_idx} is out of range (max: {len(items) - 1})")
        return

    find_similar(
        query_idx=query_idx,
        embeddings=embeddings,
        exercises=items,
        top_k=args.top_k,
        plot_tsne=args.plot_tsne,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Explore exercise embeddings and find similar items")
    p.add_argument("--config", type=str, default=None, help="Path to config file")
    p.add_argument("--dataset_path", type=str, default=None, help="Override dataset path")
    p.add_argument("--query_idx", type=int, default=0, help="Index of query exercise")
    p.add_argument("--top_k", type=int, default=10, help="Number of similar exercises to find")
    p.add_argument("--plot_tsne", action="store_true", help="Plot t-SNE visualization")
    args = p.parse_args()
    main(args)

# Usage examples:
# python src/feature_pipeline/explore_embeddings.py --config configs/feature_sentence.yaml --plot_tsne
# python src/feature_pipeline/explore_embeddings.py --config configs/feature_clip.yaml --query_idx 5 --top_k 15 --plot_tsne
# python src/feature_pipeline/explore_embeddings.py --config configs/feature.yaml --dataset_path data/processed/custom.jsonl
