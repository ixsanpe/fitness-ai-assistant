# Fitness AI Assistant

A multimodal AI system for exercise data with embeddings and semantic search. This project builds a vector database of gym exercises from images and descriptions, enabling similarity search and recommendation capabilities.

## System Architecture

```
┌─────────────────┐
│   Raw Data      │
│   (Exercises)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Feature Pipeline       │
│  - Load data            │
│  - Process images/text  │
│  - Compute embeddings   │
│    (CLIP/Sentence)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Vector Database        │
│  (Milvus)               │
│  - Store embeddings     │
│  - Similarity search    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Inference Pipeline      │
│ - Query processing      │
│ - Similarity retrieval  │
│ - Results ranking       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│  Applications   │
│  - Web API      │
│  - Recommender  │
│  - CLI Tool     │
└─────────────────┘
```

## Project Structure

- **`src/feature_pipeline/`** - Data loading and embedding computation
  - `loaders/` - Exercise data loading from raw directories
  - `embedders/` - CLIP and sentence transformer embedders
  - `storage/` - Embedding persistence layer

- **`src/inference_pipeline/`** - Query processing and vector search
  - `__main__.py` - CLI entry point for text queries
  - `create_db.py` - Build Milvus Lite vector database
  - `gradio_app.py` - Interactive web interface

- **`src/training_pipeline/`** - Model training (future expansion)

- **`configs/`** - Configuration files for different pipelines
  - `feature_clip.yaml` - CLIP embedding config
  - `feature_sentence.yaml` - Sentence transformer config
  - `inference.yaml` - Inference pipeline config

- **`data/`** - Datasets and embeddings
  - `raw/` - Original exercise data (images + metadata)
  - `processed/` - JSONL datasets and computed embeddings

## Setup

### Requirements

- **Python 3.9+**
- **`uv` package manager**

### Local Development

```bash
chmod +x infra/setup-local.sh
./infra/setup-local.sh
source .venv/bin/activate
```

This initializes the environment, installs PyTorch (with proper backend support), and downloads the exercise dataset.

## Usage

### Generate Embeddings

```bash
# Using CLIP for multimodal embeddings
python -m src.feature_pipeline --config configs/feature_clip.yaml

# Using sentence transformers for text-only
python -m src.feature_pipeline --config configs/feature_sentence.yaml

# Process limited sample
python -m src.feature_pipeline --config configs/feature_clip.yaml --sample_limit 100
```

### Query the Vector DB

```bash
python -m src.inference_pipeline --query "push up" --top_k 5
```

### Launch Interactive Interface

```bash
python src/inference_pipeline/gradio_app.py
```

### Build Vector Database (Milvus Lite)

```bash
python src/inference_pipeline/create_db.py --config configs/inference.yaml
```

## Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_feature_pipeline.py

# Run with verbose output
pytest -v
```

Tests are located in the `tests/` directory and cover:
- Configuration loading and validation
- Feature pipeline components
- Inference pipeline components

## Dataset

Uses gym exercise data from [exercises.json](https://github.com/wrkout/exercises.json). The dataset includes exercise images and metadata processed into a queryable vector format.

## Future Plans

- Training pipeline for fine-tuned embeddings
- Recommender system for personalized workout suggestions
- Training progress tracking and analytics
- Multi-language support
