import os
from pathlib import Path
from typing import List, Tuple

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import glob
import json

from src.inference_pipeline.pipeline import InferencePipeline


DATA_RAW_EXERCISES = Path("data/raw/exercises")
METADATA_PATH = Path("data/processed/exercises_dataset.jsonl")


def _load_metadata():
    """Load JSONL metadata into a list. Order should match the embeddings/idx order."""
    if not METADATA_PATH.exists():
        return []
    meta = []
    with METADATA_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                meta.append(json.loads(line))
            except Exception:
                continue
    return meta


_METADATA = _load_metadata()


def find_image_for_id(ex_id: str) -> str:
    """Return path to first image for an exercise id, or empty string if not found."""
    # common locations — try raw path and fallback to data/exercises
    candidates = []
    candidates.append(DATA_RAW_EXERCISES / ex_id / "images")
    candidates.append(Path("data/exercises") / ex_id / "images")
    for c in candidates:
        if c.exists() and c.is_dir():
            # find jpg/png files
            imgs = sorted(glob.glob(str(c / "*.jpg")) + glob.glob(str(c / "*.png")) + glob.glob(str(c / "*.jpeg")))
            if imgs:
                return imgs[0]
    return ""


def run_search(query: str, top_k: int = 5, backend: str = "local", milvus_host: str = "localhost", milvus_port: int = 19530, milvus_collection: str = "exercises_embeddings", embedding_type: str = "sentence") -> Tuple[List, List]:
    """Run pipeline query and return (images, captions) suitable for a gradio Gallery.

    Returns:
      images: list of PIL.Image or image path
      captions: list of str
    """
    # init pipeline
    try:
        pipe = InferencePipeline(backend=backend, milvus_host=milvus_host, milvus_port=milvus_port, milvus_collection=milvus_collection, embedding_type=embedding_type)
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
            idx = int(r.get("idx") if r.get("idx") is not None else (r.get("id") if isinstance(r.get("id"), int) else None))
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
        elif idx is not None and 0 <= idx < len(_METADATA):
            meta = _METADATA[idx]
            meta_name = meta.get("attributes", {}).get("name") if isinstance(meta.get("attributes"), dict) else None
            muscles = meta.get("attributes", {}).get("primaryMuscles") if isinstance(meta.get("attributes"), dict) else None
            if meta_name:
                caption = f"{meta_name} — {caption}"
            # use metadata id for folder lookup
            ex_id = meta.get("id") or ex_id

        # find image: prefer explicit image_paths in metadata (when available), else search folders
        img_path = ""
        if idx is not None and 0 <= idx < len(_METADATA):
            meta = _METADATA[idx]
            ipaths = meta.get("image_paths") or meta.get("images")
            if isinstance(ipaths, list) and ipaths:
                img_path = ipaths[0]

        if not img_path:
            img_path = find_image_for_id(str(ex_id))

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
                if isinstance(muscles, (list, tuple)):
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
        draw.text(((size[0]-w)//2, (size[1]-h)//2), text, fill=(80, 80, 80), font=font)
        return img

    if not images:
        return [], ["No results"]

    images_filled = [img if img is not None else make_placeholder() for img in images]
    return images_filled, captions


def build_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Exercise search — vector similarity\nEnter a query and choose backend (local or Milvus). Results show top-K matches and the first image for each exercise.")
        with gr.Row():
            txt = gr.Textbox(label="Query", value="sit up")
            kn = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top K")
        with gr.Row():
            backend = gr.Radio(choices=["local", "milvus"], value="local", label="Backend")
            host = gr.Textbox(label="Milvus host", value="localhost")
            port = gr.Number(label="Milvus port", value=19530)
        btn = gr.Button("Search")
        gallery = gr.Gallery(label="Results", elem_id="gallery") #.scale(grid=[3], height="auto")
        output_text = gr.Textbox(label="Info", interactive=False)

        def on_search(q, k, b, h, p):
            imgs, captions = run_search(q, int(k), backend=b, milvus_host=h, milvus_port=int(p))
            # combine into (image, caption) pairs for Gradio Gallery
            if imgs and isinstance(captions, list):
                items = list(zip(imgs, captions))
                return items, ""
            else:
                return [], "\n".join(captions if isinstance(captions, list) else [str(captions)])

        btn.click(on_search, inputs=[txt, kn, backend, host, port], outputs=[gallery, output_text])
    return demo


if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)
