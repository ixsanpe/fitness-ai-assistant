import json
from pathlib import Path
from datetime import datetime, timezone
import argparse

def folder_to_item(folder: Path):
    # Example heuristics: attributes in exercise.json or folder name
    meta_file = folder / "exercise.json"
    images_folder = folder / "images"
    attributes = {}
    if meta_file.exists():
        attributes = json.loads(meta_file.read_text(encoding="utf-8"))
    else:
        attributes["name"] = folder.name
    images = [str(p) for p in images_folder.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    def _val_to_text(v):
        if v is None:
            return ""
        if isinstance(v, (list, tuple, set)):
            return ", ".join(str(x) for x in v if x is not None)
        if isinstance(v, dict):
            # join dict values (adjust if you want keys or nested handling)
            return ", ".join(str(x) for x in v.values() if x is not None)
        return str(v)

    fields = ["name", "force", "level", "mechanic", "equipment",
              "primaryMuscles", "secondaryMuscles", "category", "instructions"]

    combined_text = " . ".join(filter(None, (_val_to_text(attributes.get(f)) for f in fields))).strip()
    return {
        "id": folder.name,
        "image_paths": images,
        "attributes": attributes,
        "combined_text": combined_text,
        "source": "wrkout-exercises-dataset",
        "created_at": datetime.now(timezone.utc).isoformat()
    }

def build_manifest(input_dir, out_file, sample_limit=None):
    #TODO: create different versions of the datasete with the limit
    input_dir = Path(input_dir)
    with open(out_file, "w", encoding="utf-8") as fo:
        for i, sub in enumerate(sorted([p for p in input_dir.iterdir() if p.is_dir()])):
            if sample_limit and i >= sample_limit:
                break
            item = folder_to_item(sub)
            fo.write(json.dumps(item, ensure_ascii=False)+"\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_file", default="dataset.jsonl")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    build_manifest(args.input_dir, args.output_file, args.limit)

# python src/data/build_dataset.py --input_dir data/raw/exercises --output_file data/processed/exercises_dataset.jsonl --limit 5