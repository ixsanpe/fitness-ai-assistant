"""Embeddings loader utilities."""

from pathlib import Path

import numpy as np

# Supported embedding types and their file patterns
EMBEDDING_PATTERNS = {
    "sentence": "sentence*",
    "clip": "clip*",
}

# Supported file formats
SUPPORTED_FORMATS = {".npy", ".pt", ".npz"}


def find_embeddings_file(embeddings_dir: Path, embedding_type: str = "sentence") -> Path | None:
    """Find an embeddings file in embeddings_dir.

    Args:
        embeddings_dir: Directory containing embeddings
        embedding_type: "sentence" for text embeddings, "clip" for multimodal

    Returns:
        Path to embeddings file or None if not found

    Raises:
        ValueError: If embedding_type is not supported
    """
    if embedding_type not in EMBEDDING_PATTERNS:
        raise ValueError(
            f"Unknown embedding_type: {embedding_type}. "
            f"Supported: {', '.join(EMBEDDING_PATTERNS.keys())}"
        )

    embeddings_dir = Path(embeddings_dir)
    if not embeddings_dir.exists():
        return None

    pattern = EMBEDDING_PATTERNS[embedding_type]
    files = list(embeddings_dir.glob(pattern))

    if not files:
        return None

    # Return the first matching file
    return sorted(files)[0]


def load_embeddings(embeddings_dir: Path, embedding_type: str = "sentence") -> np.ndarray:
    """Load embeddings from disk.

    Args:
        embeddings_dir: Directory containing embeddings
        embedding_type: "sentence" or "clip"

    Returns:
        (N, D) array of embeddings

    Raises:
        FileNotFoundError: If embeddings file not found
        ValueError: If embedding type is unsupported or file format is invalid
        RuntimeError: If embeddings cannot be loaded
    """
    embeddings_dir = Path(embeddings_dir)

    # Find embeddings file
    path = find_embeddings_file(embeddings_dir, embedding_type)
    if path is None:
        raise FileNotFoundError(
            f"No embeddings file found for type '{embedding_type}' in {embeddings_dir}"
        )

    # Validate file format
    if path.suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported embeddings format: {path.suffix}. Supported: {SUPPORTED_FORMATS}")

    # Load based on format
    try:
        if path.suffix == ".npy":
            arr = np.load(str(path))
        elif path.suffix == ".npz":
            data = np.load(str(path))
            # For .npz, assume embeddings are in 'embeddings' key
            arr = data.get("embeddings", data[list(data.files)[0]])
        elif path.suffix == ".pt":
            import torch

            tensor = torch.load(str(path))
            arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings from {path}: {e}") from e

    # Convert to float32
    arr = np.asarray(arr, dtype=np.float32)

    # Ensure 2D
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {arr.shape}")

    # Validate shape
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Empty embeddings array with shape {arr.shape}")

    print(f"✅ Loaded {embedding_type} embeddings from {path.name}")
    print(f"   Shape: {arr.shape} | Size: {arr.nbytes / 1024 / 1024:.2f} MB")

    return arr
