# Inference Pipeline

The inference pipeline runs semantic search over exercise embeddings and returns the top-K matches for a query. It supports local numpy search or Milvus-based vector search.

## Overview

The pipeline performs three main steps:

1. **Load data**: Loads embeddings and metadata (or connects to a vector database)
2. **Embed query**: Uses a text model to embed the query string
3. **Search**: Retrieves the top-K nearest exercises and returns scores and metadata

## Running the Pipeline

### Basic usage

```bash
python -m src.inference_pipeline --query "push up" --top_k 5
```

### With options

```bash
# Use a specific config
python -m src.inference_pipeline --query "sit up" --config configs/inference.yaml

# Increase result count
python -m src.inference_pipeline --query "bench press" --top_k 10
```

## Gradio UI

Launch the interactive search UI:

```bash
python src/inference_pipeline/gradio_app.py
```

With options:

```bash
python src/inference_pipeline/gradio_app.py --config configs/inference.yaml --port 8080
python src/inference_pipeline/gradio_app.py --share
```

## Vector Database Setup

When using the Milvus Lite backend, the pipeline will attempt to create the collection if initialization fails. You can also build the database explicitly:

```bash
python src/inference_pipeline/create_db.py --config configs/inference.yaml
```

## Configuration

Inference pipeline configuration includes:

- **model**: Model name and embedding type for query text
- **vector_db**: Backend selection (local, milvus, milvus_lite), collection and metric settings
- **paths**: Embeddings and model directories
- **dataset_path**: JSONL dataset path for metadata
- **device**: Device selection (auto, cuda, mps, cpu)

See `configs/inference.yaml` for an example configuration.
