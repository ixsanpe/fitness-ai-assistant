"""CLIP embedding generator."""

import torch
import tqdm
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.feature_pipeline.embedders.base import BaseEmbedder


class CLIPEmbedder(BaseEmbedder):
    """Generate CLIP embeddings (text + image)."""

    def compute(self, items: list[dict]) -> torch.Tensor:
        existing = self.should_skip()
        if existing is not None:
            return existing

        print(f"Computing CLIP embeddings with model: {self.config.embedding.model_name}")
        model = CLIPModel.from_pretrained(self.config.embedding.model_name).to(self.device)  # type: ignore[call-arg]
        processor = CLIPProcessor.from_pretrained(self.config.embedding.model_name, use_fast=True)  # type: ignore[call-arg]

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
            text_input = processor.tokenizer( # type: ignore
                [text],
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

        # Combine embeddings
        combined_embeddings = self._combine_embeddings(text_embeddings, image_embeddings)

        self.storage.save(
            combined_embeddings,
            embedding_type=self.config.embedding.output_prefix,
            save_format=self.config.embedding.save_format,  # type: ignore[arg-type]
        )
        return combined_embeddings

    def _process_images(
        self, images: list[str], processor: CLIPProcessor, model: CLIPModel
    ) -> torch.Tensor:
        """Process images and aggregate embeddings."""
        img_embs = []

        for img_path in images:
            try:
                img = Image.open(img_path).convert("RGB")
                image_input = processor(images=[img], return_tensors="pt") # type: ignore
                image_input = {k: v.to(self.device) for k, v in image_input.items()}
                with torch.no_grad():
                    img_emb = model.get_image_features(**image_input)
                img_embs.append(img_emb.cpu())
            except Exception as e:
                print(f"Warning: Failed to process image {img_path}: {e}")
                continue

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
            return torch.zeros((1, model.config.projection_dim))

    def _combine_embeddings(
        self, text_embeddings: torch.Tensor, image_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Combine text and image embeddings with weights."""
        text_weight = self.config.embedding.text_weight
        image_weight = self.config.embedding.image_weight

        combined = torch.cat(
            [text_embeddings * text_weight, image_embeddings * image_weight], dim=1
        )

        if self.config.embedding.normalize:
            combined = torch.nn.functional.normalize(combined, p=2, dim=1)

        return combined
