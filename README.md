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
┌─────────────────────────┐
│ Generation (optional)   │
│ - Local Ollama LLM      │
│ - Grounded answer over  │
│   retrieved results     │
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
  - `pipeline.py` - `InferencePipeline`: `.query()` for retrieval,
    `.answer()` for retrieval + grounded generation
  - `create_db.py` - Build Milvus Lite vector database
  - `backends/` - `SearchBackend` implementations: local (numpy), Milvus Lite,
    BM25 (`bm25.py`), plus a standalone Lance demo (`lance_demo.py`)
  - `generation.py` - `OllamaGenerator`: the "G" in RAG — generates a grounded
    answer over retrieved results via a local Ollama model
  - `gradio_app.py` - Interactive web interface, with a toggle to run the
    RAG generation step over search results

- **`src/eval/`** - Retrieval evaluation
  - `build_eval_set.py` - Bootstraps the labeled eval set + disjoint
    contrastive training pairs from exercise metadata
  - `eval_harness.py` - Metrics (precision/recall/nDCG/MRR@k) over any
    `retrieve_fn`
  - `fusion.py` - Reciprocal Rank Fusion for hybrid retrieval
  - `benchmark_retrieval.py` - BM25 vs dense vs hybrid comparison
  - `index_sweep.py` - FLAT vs IVF_FLAT vs HNSW recall/latency sweep (no
    difference at this dataset's scale — see Retrieval Quality below)

- **`src/training_pipeline/`** - Model training
  - `contrastive_finetune.py` - Fine-tunes the sentence embedder with
    `MultipleNegativesRankingLoss` on bootstrapped query/positive pairs —
    Recall@10 0.356 → 0.813 on the eval set (see Retrieval Quality below)
  - `mlflow_experiments.py` - MLflow-tracked embedding model comparisons

- **`configs/`** - Configuration files for different pipelines
  - `feature_clip.yaml` - CLIP embedding config
  - `feature_sentence.yaml` - Sentence transformer config
  - `inference.yaml` - Inference pipeline config (vector DB backend, plus an
    optional `generation:` block for RAG answer generation via Ollama)
  - `training_contrastive.yaml` - Contrastive fine-tuning config

- **`data/`** - Datasets and embeddings
  - `raw/` - Original exercise data (images + metadata)
  - `processed/` - JSONL datasets and computed embeddings
  - `eval/` - Labeled eval set (`test_queries.jsonl`)
  - `training/` - Bootstrapped contrastive training pairs

- **`results/`** - Benchmark output (metrics tables, index sweep plot)

- **`docs/`** - Design notes and write-ups (see `docs/README.md` for the
  index): `LESSONS_LEARNT.md`, `milvus_vs_lance.md`, `rag_next_steps.md`,
  `agentic_search_ladder.md`, `retrieval_metrics.html`

## Setup

### Requirements

- **Python 3.9+**
- **`uv` package manager**
- **[Ollama](https://ollama.com)** (optional, for grounded answer generation
  — run `ollama serve` and `ollama pull qwen2.5:7b`, or whatever model you
  set in `configs/inference.yaml`)

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

### Generate a Grounded Answer (RAG)

Set `generation.enabled: true` in the config (already on in
`configs/inference.yaml`), then make sure a local Ollama server is running
with that model pulled:

```bash
ollama serve &
ollama pull qwen2.5:7b

python -m src.inference_pipeline --query "good bodyweight leg exercise" --top_k 5
```

With generation enabled, the CLI retrieves results as usual and then asks the
local model to answer using only those retrieved exercises as context
(`src/inference_pipeline/generation.py`, `InferencePipeline.answer()`). If
Ollama isn't reachable, the pipeline raises a clear error rather than
silently falling back to retrieval-only.

### Run MLflow Experiments
```bash
# Compare multiple embedding models
python -m src.training_pipeline.mlflow_experiments \
  --mode compare \
  --configs configs/feature_sentence.yaml configs/feature_clip.yaml

# View results
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

### Fine-tune the Sentence Embedder (Contrastive)

```bash
python -m src.training_pipeline.contrastive_finetune --config configs/training_contrastive.yaml
```

Fine-tunes `all-MiniLM-L6-v2` with `MultipleNegativesRankingLoss` on
bootstrapped query/positive pairs and prints a before/after comparison — see
Retrieval Quality below for the numbers.

### Launch Interactive Interface

```bash
python -m src.inference_pipeline.gradio_app
```

Includes a "Generate answer (local Ollama model)" toggle that runs the RAG
generation step over the search results shown in the gallery.

### Build Vector Database (Milvus Lite)

```bash
python -m src.inference_pipeline.create_db --config configs/inference.yaml
```

### Evaluate Retrieval Quality

```bash
# Bootstrap the labeled eval set + contrastive training pairs from metadata
python -m src.eval.build_eval_set

# BM25 vs dense (sentence/CLIP) vs hybrid RRF — see "Retrieval Quality" below
python -m src.eval.benchmark_retrieval --top_k 10

# FLAT vs IVF_FLAT vs HNSW recall/latency sweep on Milvus Lite
python -m src.eval.index_sweep --top_k 10 --num_trials 200
```

## Retrieval Quality

Scored on a 69-query bootstrapped eval set (`src/eval/build_eval_set.py`),
comparing hybrid RRF (BM25 + dense), CLIP, and plain text embeddings.
Contrastive fine-tuning improves Recall@10 to 0.81, though that's mostly
domain-vocabulary fit, not general-purpose gain. An ANN index sweep
(FLAT/IVF_FLAT/HNSW) found no recall or latency benefit at this dataset's
scale. Full metrics tables land in `results/`; methodology caveats and
follow-ups are in `docs/rag_next_steps.md`.

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
- Retrieval evaluation (metrics, RRF fusion, BM25)

## Dataset

Uses gym exercise data from [exercises.json](https://github.com/wrkout/exercises.json). The dataset includes exercise images and metadata processed into a queryable vector format.

## Documentation

Deeper write-ups live in [`docs/`](docs/README.md): lessons learnt building
the pipeline, a Milvus-vs-Lance vector-store comparison, the planned RAG
roadmap, and where this project's retrieval sits on the agentic-search
maturity ladder.

## Future Plans

- Cross-encoder reranking over hybrid retrieval's top-k
- FAISS-based ANN sweep, to cross-check the Milvus Lite index sweep against a
  standalone library without the local-mode `HNSW` restriction
- Recommender system for personalized workout suggestions
- Training progress tracking and analytics
- Multi-language support
