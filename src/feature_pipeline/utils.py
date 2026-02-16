"""Shared utilities for feature pipeline."""

import json
from pathlib import Path

import numpy as np
import torch


def load_dataset(path: Path | str) -> list[dict]:
    """Load dataset from JSONL file."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items)} items from {path}")
    return items


def load_embeddings(embeddings_dir: Path, embedding_type: str) -> torch.Tensor | None:
    """Load existing embeddings if they exist."""
    pattern = f"{embedding_type}*"
    existing = list(embeddings_dir.glob(pattern))
    if existing:
        emb_path = existing[0]
        print(f"Found existing embeddings: {emb_path}")
        return torch.tensor(np.load(str(emb_path)))
    return None


def save_embeddings(embeddings: torch.Tensor, output_path: Path, save_format: str = "npy"):
    """Save embeddings to disk."""
    if save_format == "npy":
        np.save(str(output_path), embeddings.cpu().numpy())
    elif save_format == "pt":
        torch.save(embeddings, output_path)
    else:
        raise ValueError(f"Unsupported save format: {save_format}")
    print(f"Embeddings saved to {output_path}")


def combine_text_fields(attributes: dict, fields: list[str]) -> str:
    """Combine text fields into a single string."""

    def _val_to_text(v):
        if v is None:
            return ""
        if isinstance(v, list | tuple | set):
            return ", ".join(str(x) for x in v if x is not None)
        if isinstance(v, dict):
            return ", ".join(str(x) for x in v.values() if x is not None)
        return str(v)

    return " . ".join(filter(None, (_val_to_text(attributes.get(f)) for f in fields))).strip()
