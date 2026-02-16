"""Text embedder model wrapper."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """Wrapper for sentence transformer embeddings."""

    def __init__(self, model_name: str, device: str = "cpu"):
        """Initialize embedder.

        Args:
            model_name: HuggingFace model name (e.g., "all-MiniLM-L6-v2")
            device: Device to use ("cpu", "cuda", "mps", or "auto")
        """
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self._model = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve device string to actual torch device.

        Args:
            device: Device specification ("auto", "cpu", "cuda", "mps", etc.)

        Returns:
            Resolved device string
        """
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _ensure_model(self):
        """Lazy-load the model."""
        if self._model is None:
            print(f"Loading model: {self.model_name} on device: {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Input text

        Returns:
            (D,) normalized embedding vector
        """
        self._ensure_model()
        emb = self._model.encode([text], convert_to_numpy=True)[0].astype(np.float32)

        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts.

        Args:
            texts: List of input texts

        Returns:
            (N, D) normalized embedding matrix
        """
        self._ensure_model()
        embeddings = self._model.encode(texts, convert_to_numpy=True).astype(np.float32)

        # Normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        return embeddings
