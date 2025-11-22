from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import torch
import json
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from pathlib import Path
from transformers import CLIPProcessor
import tqdm

EMBEDDINGS_OUT_PATH = Path("data/processed/embeddings")
DEBUG = False

def load_dataset(path):
    items=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def load_embeddings(type: str):
    if type == "sentence":
        flag = "*text.npy"
    elif type == "clip":
        flag = "*clip.npy"
    else:
        raise ValueError(f"Unknown embedding type: {type}")
    existing = list(EMBEDDINGS_OUT_PATH.glob(flag))
    if existing:
        text_path = existing[0]
        print(f"Found embeddings file: {text_path}")
        return torch.tensor(np.load(str(text_path)))

def compute_sentence_embeddings(items, out_prefix, device, model_name="all-MiniLM-L6-v2", batch=64) -> torch.Tensor:
    # Ensure out_prefix
    if not out_prefix:
        out_prefix = "default"

    # Ensure output directory exists
    EMBEDDINGS_OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Look for any existing embeddings file ending with "text.npy"
    existing = list(EMBEDDINGS_OUT_PATH.glob("*text.npy"))
    embeddings = load_embeddings("sentence")
    if embeddings is not None:
        return embeddings

    model = SentenceTransformer(model_name, device=device)
    texts = [it.get("combined_text","") for it in items]
    embeddings = model.encode(texts, batch_size=batch, show_progress_bar=True, convert_to_tensor=True)

    # Save embeddings
    text_path = EMBEDDINGS_OUT_PATH / f"{out_prefix}_text.npy"
    np.save(str(text_path), embeddings.cpu().numpy())
    print(f"Embeddings saved to {text_path}")
    return embeddings

def compute_clip_embeddings(items, out_prefix, device, model_name="openai/clip-vit-base-patch32") -> torch.Tensor:
    
    # Look for any existing embeddings file ending with "clip.npy"
    embeddings = load_embeddings("clip")
    if embeddings is not None:
        return embeddings

    # Compute clip embeddings
    model = CLIPModel.from_pretrained(model_name).to(device)
    max_position_embeddings = model.config.text_config.max_position_embeddings
    print("Max position embeddings:", max_position_embeddings)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print(vars(processor))         # instance attributes (if available)
    """
    This is a CLIPProcessor class from the transformers library. Its arguments include:
    - CLIPImageProcessor
    - CLIPTokenizerFast
    """
    text_embeddings = []
    image_embeddings = []
    for it in tqdm(items):
        text = it.get("combined_text","")
        images = it.get("image_paths",[])
        # Process text
        text_input = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        """
        This is a BatchFeature class from the transformer library with these keys:
        - input_ids
        - attention_mask
        """
        if DEBUG:
            print("Processing text:", text_input.data["input_ids"].shape, text_input.data["attention_mask"].shape)
            raise Exception("Debugging")
        with torch.no_grad():
            text_emb = model.get_text_features(**text_input)
        text_embeddings.append(text_emb.cpu())
        # Process images
        img_embs = []
        for img_path in images:
            image_input = processor(images=[img_path], return_tensors="pt").to(device)
            with torch.no_grad():
                img_emb = model.get_image_features(**image_input)
            img_embs.append(img_emb.cpu())
        if img_embs:
            image_embeddings.append(torch.mean(torch.stack(img_embs), dim=0))
        else:
            image_embeddings.append(torch.zeros((1, model.config.projection_dim)))
    text_embeddings = torch.cat(text_embeddings, dim=0)
    image_embeddings = torch.cat(image_embeddings, dim=0)
    combined_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)
    # Save embeddings
    path = EMBEDDINGS_OUT_PATH / f"{out_prefix}_clip.npy"
    np.save(str(path), combined_embeddings.cpu().numpy())
    return combined_embeddings

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    items = load_dataset(args.dataset_path)

    if args.embed_tool == "sentence":
        embeddings = compute_sentence_embeddings(items, args.out_prefix, device)
        print("Text embeddings computed:", embeddings.shape)
    elif args.embed_tool == "clip":
        embeddings = compute_clip_embeddings(items, args.out_prefix, device)
        print("CLIP embeddings computed:", embeddings.shape)
    else:
        raise ValueError(f"Unknown embed_tool: {args.embed_tool}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--embed_tool", default="sentence")  # or "clip"
    p.add_argument("--out_prefix", default=None)
    args = p.parse_args()
    main(args)

# python src/data/compute_embeddings.py --dataset_path data/processed/exercises_dataset.jsonl --embed_tool "sentence" --out_prefix "sentence"
# python src/data/compute_embeddings.py --dataset_path data/processed/exercises_dataset.jsonl --embed_tool "clip" --out_prefix "clip"