import glob
import json
import sys
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config
from src.inference_pipeline.pipeline import InferencePipeline


def _load_metadata(metadata_path: Path):
    """Load JSONL metadata into a list. Order should match the embeddings/idx order."""
    if not metadata_path.exists():
        return []
    meta = []
    with metadata_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                meta.append(json.loads(line))
            except Exception:
                continue
    return meta


def find_image_for_id(ex_id: str, raw_exercises_dir: Path) -> str:
    """Return path to first image for an exercise id, or empty string if not found."""
    # common locations — try raw path and fallback to data/exercises
    candidates = []
    candidates.append(raw_exercises_dir / ex_id / "images")
    candidates.append(Path("data/exercises") / ex_id / "images")
    for c in candidates:
        if c.exists() and c.is_dir():
            # find jpg/png files
            imgs = sorted(
                glob.glob(str(c / "*.jpg"))
                + glob.glob(str(c / "*.png"))
                + glob.glob(str(c / "*.jpeg"))
            )
            if imgs:
                return imgs[0]
    return ""


def run_search(
    query: str,
    top_k: int = 5,
    config_path: str = None,
) -> tuple[list, list]:
    """Run pipeline query and return (images, captions) suitable for a gradio Gallery.

    Returns:
      images: list of PIL.Image or image path
      captions: list of str
    """
    # Load config and init pipeline
    try:
        config = load_config("inference", config_path)
        pipe = InferencePipeline(config)

        # Load metadata and get paths from config
        metadata_path = Path(config.dataset_path)
        raw_exercises_dir = Path(config.paths.raw_data_dir) / "exercises"
        metadata = _load_metadata(metadata_path)
    except Exception as e:
        return [], [f"Failed to initialize pipeline: {e}"]

    try:
        res = pipe.query(query, top_k=top_k)
    except Exception as e:
        return [], [f"Query failed: {e}"]
    print(f"Results for query: '{res}'")
    images = []
    captions = []

    for r in res:
        # pipeline may return numeric idxs (into the embeddings array) or string ids
        idx = None
        raw_id = r.get("id") if r.get("id") is not None else r.get("idx")
        try:
            idx = int(
                r.get("idx")
                if r.get("idx") is not None
                else (r.get("id") if isinstance(r.get("id"), int) else None)
            )
        except Exception:
            idx = None

        # default caption uses whatever id we have; we'll improve it with metadata if available
        ex_id = raw_id
        score = r.get("score")
        caption = f"{ex_id} (score={score:.4f})"

        # prefer attributes returned in the hit
        muscles = None
        if r.get("attributes") and isinstance(r.get("attributes"), dict):
            name = r["attributes"].get("name")
            muscles = r["attributes"].get("primaryMuscles")
            if name:
                caption = f"{name} — {caption}"
        # otherwise map idx -> metadata entry (if available)
        elif idx is not None and 0 <= idx < len(metadata):
            meta = metadata[idx]
            meta_name = (
                meta.get("attributes", {}).get("name")
                if isinstance(meta.get("attributes"), dict)
                else None
            )
            muscles = (
                meta.get("attributes", {}).get("primaryMuscles")
                if isinstance(meta.get("attributes"), dict)
                else None
            )
            if meta_name:
                caption = f"{meta_name} — {caption}"
            # use metadata id for folder lookup
            ex_id = meta.get("id") or ex_id

        # find image: prefer explicit image_paths in metadata (when available), else search folders
        img_path = ""
        if idx is not None and 0 <= idx < len(metadata):
            meta = metadata[idx]
            ipaths = meta.get("image_paths") or meta.get("images")
            if isinstance(ipaths, list) and ipaths:
                img_path = ipaths[0]

        if not img_path:
            img_path = find_image_for_id(str(ex_id), raw_exercises_dir)

        if img_path:
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception:
                images.append(None)
        else:
            images.append(None)

        # append muscle info to caption if available
        if muscles:
            try:
                if isinstance(muscles, list | tuple):
                    muscles_str = ", ".join(muscles)
                else:
                    muscles_str = str(muscles)
                caption = f"{caption} — muscles: {muscles_str}"
            except Exception:
                pass

        captions.append(caption)

    # Replace any None images with a light placeholder so Gradio won't raise on NoneType
    def make_placeholder(size=(320, 240), text="No image"):
        img = Image.new("RGB", size, color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        w, h = draw.textsize(text, font=font)
        draw.text(((size[0] - w) // 2, (size[1] - h) // 2), text, fill=(80, 80, 80), font=font)
        return img

    if not images:
        return [], ["No results"]

    images_filled = [img if img is not None else make_placeholder() for img in images]
    return images_filled, captions


def build_interface(default_config: str = None):
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Exercise Search — Vector Similarity\nEnter a query to find similar exercises. Results show top-K matches with images."
        )
        with gr.Row():
            txt = gr.Textbox(
                label="Query", value="sit up", placeholder="Enter exercise description..."
            )
            kn = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top K")
        with gr.Row():
            config_path = gr.Textbox(
                label="Config Path (optional)",
                value=default_config or "",
                placeholder="configs/inference.yaml or configs/inference_prod.yaml",
            )
        btn = gr.Button("🔍 Search", variant="primary")
        gallery = gr.Gallery(label="Results", elem_id="gallery", columns=3, height="auto")
        output_text = gr.Textbox(label="Info", interactive=False)

        def on_search(q, k, cfg):
            cfg_path = cfg.strip() if cfg and cfg.strip() else None
            imgs, captions = run_search(q, int(k), config_path=cfg_path)
            # combine into (image, caption) pairs for Gradio Gallery
            if imgs and isinstance(captions, list):
                items = list(zip(imgs, captions, strict=False))
                return items, f"✅ Found {len(items)} results"
            else:
                return [], "\n".join(captions if isinstance(captions, list) else [str(captions)])

        btn.click(on_search, inputs=[txt, kn, config_path], outputs=[gallery, output_text])
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch Gradio exercise search interface")
    parser.add_argument(
        "--config", type=str, default="configs/inference.yaml", help="Path to inference config file"
    )
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    args = parser.parse_args()

    app = build_interface(default_config=args.config)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)

# Usage:
# python src/inference_pipeline/gradio_app.py
# python src/inference_pipeline/gradio_app.py --config configs/inference.yaml
# python src/inference_pipeline/gradio_app.py --config configs/inference_prod.yaml --port 8080
# python src/inference_pipeline/gradio_app.py --share  # Creates public URL
