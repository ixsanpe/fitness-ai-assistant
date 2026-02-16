"""Exercise dataset loader."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

from src.feature_pipeline.loaders.base import BaseLoader
from src.feature_pipeline.utils import combine_text_fields


class ExerciseLoader(BaseLoader):
    """Loader for exercise dataset from folder structure."""

    IMAGE_EXTENSIONS: ClassVar[set[str]] = {".jpg", ".jpeg", ".png", ".webp"}
    REQUIRED_FIELDS: ClassVar[list[str]] = [
        "name",
        "force",
        "level",
        "mechanic",
        "equipment",
        "primaryMuscles",
        "secondaryMuscles",
        "category",
        "instructions",
    ]

    def __init__(
        self,
        min_text_length: int = 10,
        remove_duplicates: bool = True,
        sample_limit: int | None = None,
        progress_interval: int = 100,
    ):
        """Initialize the exercise loader.

        Args:
            min_text_length: Minimum length for combined text to be included
            remove_duplicates: Whether to remove duplicate text entries
            sample_limit: Maximum number of exercises to load (None for all)
            progress_interval: Report progress every N items
        """
        self.min_text_length = min_text_length
        self.remove_duplicates = remove_duplicates
        self.sample_limit = sample_limit
        self.progress_interval = progress_interval

    def load(self, path: Path) -> list[dict]:
        """Load exercises from directory structure.

        Args:
            path: Directory containing exercise folders

        Returns:
            List of exercise items with metadata and image paths

        Raises:
            FileNotFoundError: If the path doesn't exist
            NotADirectoryError: If the path is not a directory
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Exercise directory not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")

        items = []
        seen_texts = set() if self.remove_duplicates else None
        skipped_count = {"too_short": 0, "duplicate": 0, "error": 0}

        # Use generator to avoid loading all folders into memory
        folders = (p for p in path.iterdir() if p.is_dir())

        for i, folder in enumerate(folders):
            if self.sample_limit and i >= self.sample_limit:
                print(f"Reached sample limit of {self.sample_limit}")
                break

            if i > 0 and i % self.progress_interval == 0:
                print(f"Processed {i} folders, loaded {len(items)} items")

            try:
                item = self._folder_to_item(folder)
            except Exception as e:
                print(f"Warning: Failed to process folder {folder.name}: {e}")
                skipped_count["error"] += 1
                continue

            combined_text = item.get("combined_text", "")

            if len(combined_text) < self.min_text_length:
                skipped_count["too_short"] += 1
                continue

            if self.remove_duplicates:
                if combined_text in seen_texts:
                    skipped_count["duplicate"] += 1
                    continue
                seen_texts.add(combined_text) # type: ignore

            items.append(item)

        print(
            f"Loaded {len(items)} items. "
            f"Skipped: {skipped_count['too_short']} too short, "
            f"{skipped_count['duplicate']} duplicates, "
            f"{skipped_count['error']} errors"
        )
        return items

    def _folder_to_item(self, folder: Path) -> dict:
        """Convert exercise folder to dataset item.

        Args:
            folder: Path to exercise folder

        Returns:
            Dictionary with exercise data

        Raises:
            json.JSONDecodeError: If exercise.json is malformed
        """
        meta_file = folder / "exercise.json"
        images_folder = folder / "images"

        # Load attributes from JSON or use folder name as fallback
        if meta_file.exists():
            try:
                attributes = json.loads(meta_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in {meta_file}: {e}")
                attributes = {"name": folder.name}
        else:
            attributes = {"name": folder.name}

        # Collect valid image paths
        image_paths = []
        if images_folder.exists() and images_folder.is_dir():
            image_paths = [
                p
                for p in images_folder.iterdir()
                if p.is_file() and p.suffix.lower() in self.IMAGE_EXTENSIONS
            ]

        # Sort for deterministic output
        image_paths.sort()

        combined_text = combine_text_fields(attributes, self.REQUIRED_FIELDS)

        return {
            "id": folder.name,
            "image_paths": [str(p) for p in image_paths],
            "attributes": attributes,
            "combined_text": combined_text,
            "source": str(folder),
            "created_at": datetime.now(UTC).isoformat(),
        }

    def save(self, items: list[dict], output_path: Path) -> int:
        """Save items to JSONL file.

        Args:
            items: List of exercise items to save
            output_path: Path where JSONL file should be written

        Returns:
            Number of items saved

        Raises:
            OSError: If file cannot be written
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"Error: Failed to write to {output_path}: {e}")
            raise

        print(f"Saved {len(items)} items to {output_path}")
        return len(items)
