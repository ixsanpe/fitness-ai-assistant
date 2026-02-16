"""Similarity visualization for embeddings."""

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


def find_similar(
    query_idx: int,
    embeddings: torch.Tensor,
    exercises: list[dict],
    top_k: int = 10,
    plot_tsne: bool = False,
):
    """Find similar exercises based on embedding similarity."""
    if query_idx >= len(exercises):
        raise ValueError(f"query_idx {query_idx} out of range (max: {len(exercises) - 1})")
    if embeddings.shape[0] != len(exercises):
        raise ValueError(
            f"Embeddings ({embeddings.shape[0]}) and exercises ({len(exercises)}) length mismatch"
        )

    query_emb = embeddings[query_idx].unsqueeze(0)
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
        plot_tsne_visualization(embeddings, exercises, query_idx, similar_idx.tolist())


def plot_tsne_visualization(
    embeddings: torch.Tensor,
    exercises: list[dict],
    query_idx: int,
    similar_idx: list[int],
):
    """Plot t-SNE visualization of embeddings."""
    embeddings_np = embeddings.cpu().numpy()

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
