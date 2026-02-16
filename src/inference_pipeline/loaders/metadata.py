"""Metadata loader for exercise datasets."""

import json
from pathlib import Path


def load_metadata(dataset_path: Path) -> list[dict]:
    """Load JSONL metadata file.

    Args:
        dataset_path: Path to JSONL dataset file

    Returns:
        List of metadata dicts with keys: id, combined_text, attributes, etc.
    """
    items = []
    dataset_path = Path(dataset_path)

    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"✅ Loaded {len(items)} metadata items from {dataset_path.name}")
    return items
