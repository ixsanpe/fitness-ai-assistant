import glob
import json
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from src.config import load_config
from src.config.inference import InferenceConfig
from src.inference_pipeline.pipeline import InferencePipeline

# Cache pipelines by config path — building one opens a Milvus-lite connection that
# is never truly released (MilvusBackend.close() is a no-op), so rebuilding it on
# every request causes the next request to fail to open the locked db file and fall
# back to dropping/recreating the whole collection.
_pipeline_cache: dict[str, InferencePipeline] = {}


def _get_pipeline(config_path: str) -> InferencePipeline:
    if config_path not in _pipeline_cache:
        config = load_config("inference", config_path)
        assert isinstance(config, InferenceConfig), "Config must be an InferenceConfig instance"
        _pipeline_cache[config_path] = InferencePipeline(config)
    return _pipeline_cache[config_path]


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
    config_path: str | None = None,
    generate_answer: bool = False,
) -> tuple[list, list, str]:
    """Run pipeline query and return (images, captions, answer) for the gradio UI.

    Returns:
      images: list of PIL.Image or image path
      captions: list of str
      answer: generated answer text (empty string if generation wasn't requested)
    """
    # Load (cached) pipeline
    try:
        resolved_config_path = config_path or "configs/inference.yaml"
        pipe = _get_pipeline(resolved_config_path)

        # Load metadata and get paths from config
        metadata_path = Path(pipe.config.dataset_path)
        raw_exercises_dir = Path(pipe.config.paths.raw_data_dir) / "exercises"
        metadata = _load_metadata(metadata_path)
    except Exception as e:
        return [], [f"Failed to initialize pipeline: {e}"], ""

    try:
        res = pipe.query(query, top_k=top_k)
    except Exception as e:
        return [], [f"Query failed: {e}"], ""

    answer = ""
    if generate_answer:
        if pipe.generator is None:
            answer = "⚠️ Generation not enabled — set generation.enabled: true in the config."
        else:
            try:
                answer = pipe.generator.generate(query, res)
            except Exception as e:
                answer = f"⚠️ Generation failed: {e}"

    print(f"Results for query: '{res}'")
    images = []
    captions = []

    for r in res:
        # pipeline may return numeric idxs (into the embeddings array) or string ids
        idx = None
        raw_id = r.get("id") if r.get("id") is not None else r.get("idx")
        try:
            idx_val = (
                r.get("idx")
                if r.get("idx") is not None
                else (r.get("id") if isinstance(r.get("id"), int) else None)
            )
            idx = int(idx_val) if idx_val is not None else None
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
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(((size[0] - w) // 2, (size[1] - h) // 2), text, fill=(80, 80, 80), font=font)
        return img

    if not images:
        return [], ["No results"], answer

    images_filled = [img if img is not None else make_placeholder() for img in images]
    return images_filled, captions, answer


def build_interface(default_config: str | None = None):
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
            gen_toggle = gr.Checkbox(
                label="Generate answer (local Ollama model)",
                value=False,
            )
        btn = gr.Button("🔍 Search", variant="primary")
        answer_box = gr.Textbox(label="Answer", interactive=False, visible=False, lines=4)
        gallery = gr.Gallery(label="Results", elem_id="gallery", columns=3, height="auto")
        output_text = gr.Textbox(label="Info", interactive=False)

        def on_search(q, k, cfg, gen):
            cfg_path = cfg.strip() if cfg and cfg.strip() else None
            imgs, captions, answer = run_search(
                q, int(k), config_path=cfg_path, generate_answer=gen
            )
            # combine into (image, caption) pairs for Gradio Gallery
            if imgs and isinstance(captions, list):
                items = list(zip(imgs, captions, strict=False))
                info = f"✅ Found {len(items)} results"
            else:
                items = []
                info = "\n".join(captions if isinstance(captions, list) else [str(captions)])
            return items, info, gr.update(value=answer, visible=bool(gen))

        btn.click(
            on_search,
            inputs=[txt, kn, config_path, gen_toggle],
            outputs=[gallery, output_text, answer_box],
        )
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
# python -m src.inference_pipeline.gradio_app
# python -m src.inference_pipeline.gradio_app --config configs/inference.yaml
# python -m src.inference_pipeline.gradio_app --config configs/inference_prod.yaml --port 8080
# python -m src.inference_pipeline.gradio_app --share  # Creates public URL
