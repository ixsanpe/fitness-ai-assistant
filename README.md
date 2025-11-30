# Fitness AI Assistant

A gym training assistant using embeddings and vector search for exercise recommendations. Built with MLflow experiment tracking, multi-modal embeddings (text + images), and Milvus vector database.

## Project Structure
The project follows the FTI architecture.
```
fitness-ai-assistant/
├── configs/              # YAML configs for features, training, and inference
├── data/
│   ├── raw/             # Exercise images and metadata
│   ├── processed/       # JSONL datasets and embeddings
│   └── vector_db/       # Milvus vector database
├── src/
│   ├── feature_pipeline/     # Dataset building and embedding generation
│   ├── training_pipeline/    # MLflow experiments and model comparison
│   ├── inference_pipeline/   # Query interface and Gradio app
│   └── config/              # Configuration management
└── tests/               # Unit and integration tests
```

## Quick Start

### Prerequisites
- Python 3.11+
- pipx (for uv installation)
- Docker (optional, for Milvus vector database)
- CUDA-compatible GPU (optional, for faster embedding generation)

### Setup Environment
```bash
# Install uv package manager
pipx install uv

# Initialize project and download dataset
chmod +x infra/setup-local.sh
./infra/setup-local.sh

# Activate virtual environment
source .venv/bin/activate
```

### Run the Pipeline

**1. Build Dataset**
```bash
python src/feature_pipeline/build_dataset.py \
  --config configs/feature_sentence.yaml
```
Processes raw exercise data into JSONL format with combined text fields.

**2. Generate Embeddings**
```bash
# Sentence embeddings
python src/feature_pipeline/compute_embeddings.py \
  --config configs/feature_sentence.yaml

# CLIP embeddings (multimodal)
python src/feature_pipeline/compute_embeddings.py \
  --config configs/feature_clip.yaml
```

**3. Run MLflow Experiments**
```bash
# Compare multiple embedding models
python src/training_pipeline/mlflow_experiments.py \
  --mode compare \
  --configs configs/feature_sentence.yaml configs/feature_clip.yaml

# View results
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

**4. Create Vector Database**
```bash
python src/inference_pipeline/create_db.py \
  --config configs/inference.yaml
```

**5. Query Exercises**
```bash
# Command-line demo
python src/inference_pipeline/demo_query.py \
  --config configs/inference.yaml \
  --query "shoulder exercises with dumbbells"

# Gradio web interface
python src/inference_pipeline/gradio_app.py \
  --config configs/inference.yaml
```

## Key Features

- **Multi-modal embeddings**: Sentence-transformers (text) and CLIP (text + images)
- **MLflow tracking**: Compare embedding models with automatic metrics logging
- **Vector search**: Milvus-powered semantic search for exercises
- **Configurable**: YAML-based configuration for all pipelines
- **Testing**: Unit tests with pytest for all components

## Configuration

Edit configs in `configs/` to customize:
- `feature_*.yaml`: Dataset processing and embedding models
- `training.yaml`: MLflow experiment settings
- `inference*.yaml`: Vector DB and query parameters

## Development

```bash
# Run tests
pytest tests/

# Run specific test file
pytest tests/test_feature_pipeline.py -v

# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/
```

## Roadmap

🚧 **In Development:**
- Create full end-to-end pipeline
- Improve model evaluation and selection: metrics
- Add CLIP to ``mlflow`` experiments `compare` mode
- Assistant & Agent extensions (Add MCP Servers, ie Notion, and/or AI)
- Improve logging
- CI/CD and remote deployment (Docker, cloud)
