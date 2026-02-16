# Feature Pipeline

The feature pipeline is responsible for loading exercise data and computing embeddings using different models (CLIP for multimodal or sentence embeddings for text-only).

## Overview

The pipeline performs two main steps:

1. **Data Loading**: Loads exercise data from a directory structure (or uses an existing JSONL dataset) and prepares it by combining text descriptions and image paths
2. **Embedding Generation**: Computes embeddings using a configurable model (CLIP, sentence transformers, etc.)

The embeddings are stored in the configured output format (numpy arrays or PyTorch tensors) for use in downstream tasks like similarity search and inference.

## Running the Pipeline

### Basic usage

```bash
python -m src.feature_pipeline --config configs/feature_clip.yaml
```

### With options

```bash
# Process a limited sample
python -m src.feature_pipeline --config configs/feature_clip.yaml --sample_limit 100

# Use a specific dataset file
python -m src.feature_pipeline --config configs/feature_clip.yaml --dataset_path data/processed/exercises_dataset.jsonl

# Use a different input directory
python -m src.feature_pipeline --config configs/feature_clip.yaml --input_dir data/raw/exercises
```

## Configuration

Feature pipeline configuration includes:

- **embedding**: Model selection (CLIP, sentence), batch size, normalization, output format
- **dataset**: Input/output paths, text filtering, deduplication
- **paths**: Directory structure for data and models
- **device**: Device selection (auto, cuda, mps, cpu)

See `configs/feature_clip.yaml` for an example configuration.
