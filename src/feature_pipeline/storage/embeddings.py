"""Centralized embedding storage."""

from pathlib import Path
from typing import Literal

import numpy as np
import torch


class EmbeddingStorage:
    """Handle loading and saving of embeddings."""

    SUPPORTED_FORMATS = {"npy", "pt"}

    def __init__(self, embeddings_dir: Path, device: str = "cpu"):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def _find_embedding_file(self, embedding_type: str) -> Path | None:
        """Find embedding file with the given type prefix."""
        for fmt in self.SUPPORTED_FORMATS:
            path = self.embeddings_dir / f"{embedding_type}_embed.{fmt}"
            if path.exists():
                return path
        return None

    def load(self, embedding_type: str) -> torch.Tensor | None:
        """Load existing embeddings if they exist.

        Args:
            embedding_type: The type/name of embeddings to load

        Returns:
            Loaded embeddings tensor on the configured device, or None if not found
        """
        emb_path = self._find_embedding_file(embedding_type)

        if emb_path is None:
            return None

        print(f"Loading embeddings from: {emb_path}")

        if emb_path.suffix == ".npy":
            embeddings = torch.from_numpy(np.load(emb_path))
        elif emb_path.suffix == ".pt":
            embeddings = torch.load(emb_path, weights_only=True)
        else:
            raise ValueError(f"Unsupported file format: {emb_path.suffix}")

        return embeddings.to(self.device)

    def save(
        self,
        embeddings: torch.Tensor,
        embedding_type: str,
        save_format: Literal["npy", "pt"] = "npy",
    ) -> Path:
        """Save embeddings to disk.

        Args:
            embeddings: Tensor to save
            embedding_type: Name/type identifier for the embeddings
            save_format: Format to save in ('npy' or 'pt')

        Returns:
            Path where embeddings were saved

        Raises:
            ValueError: If save_format is not supported
        """
        if save_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported save format: {save_format}. "
                f"Must be one of {self.SUPPORTED_FORMATS}"
            )

        output_path = self.embeddings_dir / f"{embedding_type}_embed.{save_format}"

        if save_format == "npy":
            np.save(output_path, embeddings.cpu().numpy())
        else:  # save_format == "pt"
            torch.save(embeddings.cpu(), output_path)

        print(f"Embeddings saved to {output_path}")
        return output_path

    def exists(self, embedding_type: str) -> bool:
        """Check if embeddings exist for the given type."""
        return self._find_embedding_file(embedding_type) is not None

    def delete(self, embedding_type: str) -> bool:
        """Delete embeddings for the given type.

        Returns:
            True if file was deleted, False if it didn't exist
        """
        emb_path = self._find_embedding_file(embedding_type)
        if emb_path:
            emb_path.unlink()
            print(f"Deleted embeddings: {emb_path}")
            return True
        return False
