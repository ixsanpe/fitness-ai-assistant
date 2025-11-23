import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import tqdm
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FeatureConfig, load_config

# TODO: Use more constants (ie, embeding, dataset organiz)


def load_dataset(path):
    """Load dataset from JSONL file."""
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    print(f"Loaded {len(items)} items from {path}")
    return items


def load_embeddings(embeddings_dir: Path, embedding_type: str):
    """Load existing embeddings if they exist."""
    if embedding_type == "sentence":
        flag = "sentence*"
    elif embedding_type == "clip":
        flag = "clip*"
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    existing = list(embeddings_dir.glob(flag))
    if existing:
        emb_path = existing[0]
        print(f"Found existing embeddings: {emb_path}")
        return torch.tensor(np.load(str(emb_path)))
    return None


class EmbeddingGenerator:
    """Base class for generating embeddings from dataset items."""

    def __init__(self, config: FeatureConfig, device: str):
        """Initialize the embedding generator.

        Args:
            config: Feature configuration
            device: Device to use (cpu, cuda, etc.)
        """
        self.config = config
        self.device = device
        self.embeddings_dir = config.paths.embeddings_dir
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def load_existing(self) -> torch.Tensor | None:
        """Load existing embeddings if available and configured to skip.

        Returns:
            Tensor of embeddings if found and should be reused, None otherwise
        """
        if self.config.skip_if_exists and not self.config.overwrite:
            embeddings = load_embeddings(self.embeddings_dir, self.config.embedding.embedding_type)
            if embeddings is not None:
                print("Using existing embeddings (skip_if_exists=True)")
                return embeddings
        return None

    def compute(self, items: list[dict]) -> torch.Tensor:
        """Compute embeddings based on configured type.

        Args:
            items: List of dataset items

        Returns:
            Tensor of embeddings

        Raises:
            ValueError: If embedding type is unknown
        """
        # Check for existing embeddings first
        existing = self.load_existing()
        if existing is not None:
            return existing

        # Compute new embeddings based on type
        if self.config.embedding.embedding_type == "sentence":
            return self._compute_sentence(items)
        elif self.config.embedding.embedding_type == "clip":
            return self._compute_clip(items)
        else:
            raise ValueError(f"Unknown embedding type: {self.config.embedding.embedding_type}")

    def _compute_sentence(self, items: list[dict]) -> torch.Tensor:
        """Compute sentence embeddings using SentenceTransformer.

        Args:
            items: List of dataset items

        Returns:
            Tensor of sentence embeddings
        """
        print(f"Computing sentence embeddings with model: {self.config.embedding.model_name}")
        model = SentenceTransformer(self.config.embedding.model_name, device=self.device)
        texts = [it.get("combined_text", "") for it in items]

        embeddings = model.encode(
            texts,
            batch_size=self.config.embedding.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=self.config.embedding.normalize,
        )

        self._save_embeddings(embeddings, suffix="embed")
        return embeddings

    def _compute_clip(self, items: list[dict]) -> torch.Tensor:
        """Compute CLIP embeddings (text + image).

        Args:
            items: List of dataset items

        Returns:
            Tensor of combined CLIP embeddings
        """
        print(f"Computing CLIP embeddings with model: {self.config.embedding.model_name}")
        model = CLIPModel.from_pretrained(self.config.embedding.model_name).to(self.device)
        processor = CLIPProcessor.from_pretrained(self.config.embedding.model_name)

        max_length = (
            self.config.embedding.max_length or model.config.text_config.max_position_embeddings
        )
        print(f"Max sequence length: {max_length}")

        text_embeddings = []
        image_embeddings = []

        for it in tqdm.tqdm(items, desc="Computing CLIP embeddings"):
            text = it.get("combined_text", "")
            images = it.get("image_paths", [])

            # Process text
            text_input = processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(self.device)

            with torch.no_grad():
                text_emb = model.get_text_features(**text_input)
            text_embeddings.append(text_emb.cpu())

            # Process images
            img_emb = self._process_images(images, processor, model)
            image_embeddings.append(img_emb)

        text_embeddings = torch.cat(text_embeddings, dim=0)
        image_embeddings = torch.cat(image_embeddings, dim=0)

        # Combine embeddings with weights
        combined_embeddings = self._combine_embeddings(text_embeddings, image_embeddings)

        self._save_embeddings(combined_embeddings, suffix="embed")
        return combined_embeddings

    def _process_images(
        self, images: list[str], processor: CLIPProcessor, model: CLIPModel
    ) -> torch.Tensor:
        """Process images and aggregate embeddings.

        Args:
            images: List of image paths
            processor: CLIP processor
            model: CLIP model

        Returns:
            Aggregated image embedding tensor
        """
        img_embs = []
        max_images = self.config.dataset.max_images_per_item

        for img_path in images[:max_images] if max_images else images:
            try:
                image_input = processor(images=[img_path], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    img_emb = model.get_image_features(**image_input)
                img_embs.append(img_emb.cpu())
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

        # Aggregate based on configured mode
        if img_embs:
            img_stack = torch.stack(img_embs)
            mode = self.config.embedding.image_processing_mode

            if mode == "average":
                return torch.mean(img_stack, dim=0)
            elif mode == "first":
                return img_embs[0]
            elif mode == "max_pool":
                return torch.max(img_stack, dim=0)[0]
            else:
                return torch.mean(img_stack, dim=0)
        else:
            # Return zero embedding if no images
            return torch.zeros((1, model.config.projection_dim))

    def _combine_embeddings(
        self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Combine text and image embeddings with configured weights.

        Args:
            text_embeddings: Tensor of text embeddings
            image_embeddings: Tensor of image embeddings

        Returns:
            Combined embedding tensor
        """
        text_weight = self.config.embedding.text_weight
        image_weight = self.config.embedding.image_weight

        combined = torch.cat(
            [text_embeddings * text_weight, image_embeddings * image_weight], dim=1
        )

        if self.config.embedding.normalize:
            combined = torch.nn.functional.normalize(combined, p=2, dim=1)

        return combined

    def _save_embeddings(self, embeddings: torch.Tensor, suffix: str):
        """Save embeddings to disk.

        Args:
            embeddings: Tensor of embeddings to save
            suffix: Suffix for the filename (e.g., 'text', 'clip')
        """
        output_prefix = self.config.embedding.output_prefix
        save_format = self.config.embedding.save_format

        if save_format == "npy":
            output_path = self.embeddings_dir / f"{output_prefix}_{suffix}.npy"
            np.save(str(output_path), embeddings.cpu().numpy())
        elif save_format == "pt":
            output_path = self.embeddings_dir / f"{output_prefix}_{suffix}.pt"
            torch.save(embeddings, output_path)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")

        print(f"Embeddings saved to {output_path}")


def main(args):
    """Main function to compute embeddings."""
    # Load configuration
    config: FeatureConfig = load_config("feature", args.config)

    # Determine device
    if config.device.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device.device

    print(f"Using device: {device}")
    print(f"Embedding type: {config.embedding.embedding_type}")
    print(f"Model: {config.embedding.model_name}")

    # Load dataset
    dataset_path = args.dataset_path or config.dataset.output_path
    items = load_dataset(dataset_path)

    # Ensure output directory exists
    config.ensure_directories()

    # Create embedding generator and compute embeddings
    generator = EmbeddingGenerator(config, device)
    embeddings = generator.compute(items)

    print(f"Embeddings computed: {embeddings.shape}")
    print("✅ Embedding computation completed successfully!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute embeddings for exercise dataset")
    p.add_argument("--config", type=str, default=None, help="Path to config file")
    p.add_argument("--dataset_path", type=str, default=None, help="Override dataset path")
    args = p.parse_args()
    main(args)

# Usage examples:
# python src/feature_pipeline/compute_embeddings.py --config configs/feature_sentence.yaml
# python src/feature_pipeline/compute_embeddings.py --config configs/feature_clip.yaml
# python src/feature_pipeline/compute_embeddings.py --config configs/feature.yaml --dataset_path data/processed/custom_dataset.jsonl
