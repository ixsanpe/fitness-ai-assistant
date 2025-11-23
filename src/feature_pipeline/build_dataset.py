import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FeatureConfig, load_config


def folder_to_item(folder: Path):
    # Example heuristics: attributes in exercise.json or folder name
    meta_file = folder / "exercise.json"
    images_folder = folder / "images"
    attributes = {}
    if meta_file.exists():
        attributes = json.loads(meta_file.read_text(encoding="utf-8"))
    else:
        attributes["name"] = folder.name
    images = [
        str(p) for p in images_folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    def _val_to_text(v):
        if v is None:
            return ""
        if isinstance(v, list | tuple | set):
            return ", ".join(str(x) for x in v if x is not None)
        if isinstance(v, dict):
            # join dict values (adjust if you want keys or nested handling)
            return ", ".join(str(x) for x in v.values() if x is not None)
        return str(v)

    fields = [
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

    combined_text = " . ".join(
        filter(None, (_val_to_text(attributes.get(f)) for f in fields))
    ).strip()
    return {
        "id": folder.name,
        "image_paths": images,
        "attributes": attributes,
        "combined_text": combined_text,
        "source": "wrkout-exercises-dataset",
        "created_at": datetime.now(UTC).isoformat(),
    }


def build_dataset(
    input_dir, out_file, sample_limit=None, min_text_length=10, remove_duplicates=True
):
    """Build dataset manifest from input directory.

    Args:
        input_dir: Input directory containing exercise folders
        out_file: Output JSONL file path
        sample_limit: Maximum number of items to process
        min_text_length: Minimum text length to include item
        remove_duplicates: Remove duplicate items based on text
    """
    input_dir = Path(input_dir)
    seen_texts = set()
    items_written = 0

    with open(out_file, "w", encoding="utf-8") as fo:
        for i, sub in enumerate(sorted([p for p in input_dir.iterdir() if p.is_dir()])):
            if sample_limit and i >= sample_limit:
                break

            item = folder_to_item(sub)

            # Apply filters
            combined_text = item.get("combined_text", "")
            if len(combined_text) < min_text_length:
                continue

            if remove_duplicates:
                if combined_text in seen_texts:
                    continue
                seen_texts.add(combined_text)

            fo.write(json.dumps(item, ensure_ascii=False) + "\n")
            items_written += 1

    print(f"Built dataset with {items_written} items")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build exercise dataset from raw data")
    p.add_argument("--config", type=str, default=None, help="Path to config file")
    p.add_argument("--input_dir", type=str, default=None, help="Override input directory")
    p.add_argument("--output_file", type=str, default=None, help="Override output file")
    p.add_argument("--limit", type=int, default=None, help="Limit number of items")
    args = p.parse_args()

    # Load config
    config: FeatureConfig = load_config("feature", args.config)

    # Use config values or command-line overrides
    input_dir = args.input_dir or config.dataset.input_path
    output_file = args.output_file or config.dataset.output_path
    sample_limit = args.limit

    print(f"Building dataset from: {input_dir}")
    print(f"Output file: {output_file}")
    print(f"Min text length: {config.dataset.min_text_length}")
    print(f"Remove duplicates: {config.dataset.remove_duplicates}")

    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    build_dataset(
        input_dir=input_dir,
        out_file=output_file,
        sample_limit=sample_limit,
        min_text_length=config.dataset.min_text_length,
        remove_duplicates=config.dataset.remove_duplicates,
    )

# Usage examples:
# python src/feature_pipeline/build_dataset.py --config configs/feature_sentence.yaml
# python src/feature_pipeline/build_dataset.py --config configs/feature_sentence.yaml --limit 100
# python src/feature_pipeline/build_dataset.py --input_dir data/raw/exercises --output_file data/processed/exercises_dataset.jsonl
