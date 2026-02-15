# Configuration System Documentation

A production-ready configuration management system for the `fitness-ai-assistant` project using Pydantic for type validation and YAML for human-readable configs.

## Quick Start

### Loading Configurations

```python
from src.config import load_config

# Load from YAML file
config = load_config("inference", "configs/inference.yaml")

# Load with auto-detection (searches common paths)
config = load_config("inference")

# Load with overrides
config = load_config("inference", top_k=10)

# Access nested config
print(config.model.model_name)
print(config.retrieval.top_k)
```

### Using ConfigLoader

```python
from src.config import ConfigLoader, InferenceConfig

# Load from YAML
config = ConfigLoader.from_yaml("configs/inference.yaml", InferenceConfig)

# Load from JSON
config = ConfigLoader.from_json("configs/inference.json", InferenceConfig)

# Save to YAML
ConfigLoader.to_yaml(config, "output/config.yaml")

# Load from dictionary
config_dict = {"name": "my_config", "model": {"model_name": "my-model"}}
config = ConfigLoader.from_dict(config_dict, InferenceConfig)
```

### Creating Configs Programmatically

```python
from src.config import InferenceConfig, ModelConfig, RetrievalConfig

# Create with defaults
config = InferenceConfig()

# Create with custom values
config = InferenceConfig(
    name="my_inference",
    model=ModelConfig(
        model_name="all-MiniLM-L6-v2",
        batch_size=64
    ),
    retrieval=RetrievalConfig(
        top_k=10,
        min_score=0.7
    )
)

# Access and modify
config.retrieval.top_k = 5
print(config.model_dump())  # Convert to dict
```

## Configuration Types

### 1. Feature Pipeline Config (`FeatureConfig`)

For data processing and embedding generation:

```python
from src.config import FeatureConfig

config = FeatureConfig(
    embedding=EmbeddingConfig(
        embedding_type="sentence",
        model_name="all-MiniLM-L6-v2",
        batch_size=64
    ),
    dataset=DatasetConfig(
        input_path="data/raw/exercises",
        output_path="data/processed/exercises_dataset.jsonl"
    )
)
```

**Key Settings:**
- `embedding`: Embedding model configuration
- `dataset`: Dataset processing settings
- `skip_if_exists`: Skip processing if output exists
- `cache_embeddings`: Cache embeddings to disk

### 2. Inference Pipeline Config (`InferenceConfig`)

For model serving and similarity search:

```python
from src.config import InferenceConfig

config = InferenceConfig(
    model=ModelConfig(
        model_name="all-MiniLM-L6-v2",
        batch_size=32
    ),
    vector_db=VectorDBConfig(
        backend="milvus",
        collection_name="exercises_embeddings"
    ),
    retrieval=RetrievalConfig(
        top_k=5,
        reranking_strategy="cross_encoder"
    )
)
```

**Key Settings:**
- `model`: Embedding model for queries
- `vector_db`: Vector database backend configuration
- `retrieval`: Search and ranking parameters
- `caching`: Query result caching

### 3. Training Pipeline Config (`TrainingConfig`)

For model training and fine-tuning:

```python
from src.config import TrainingConfig

config = TrainingConfig(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    strategy="fine_tune",
    num_epochs=3,
    optimizer=OptimizerConfig(
        optimizer_type="adamw",
        learning_rate=2e-5
    ),
    data=DataConfig(
        batch_size=32,
        train_path="data/processed/train.jsonl"
    )
)
```

**Key Settings:**
- `strategy`: Training strategy (fine_tune, contrastive, etc.)
- `optimizer`: Optimizer configuration
- `scheduler`: Learning rate scheduler
- `checkpoint`: Model checkpointing settings

## Advanced Features

### Environment Variable Overrides

Override config values using environment variables:

```bash
# Format: FITNESS_AI_<SECTION>__<KEY>=value
export FITNESS_AI_MODEL__MODEL_NAME=custom-model
export FITNESS_AI_RETRIEVAL__TOP_K=10
export FITNESS_AI_DEVICE__DEVICE=cuda

# For nested values, use double underscore
export FITNESS_AI_VECTOR_DB__HOST=192.168.1.100
export FITNESS_AI_VECTOR_DB__PORT=19530
```

### Device Configuration

Automatic device selection:

```python
from src.config import DeviceConfig, DeviceType

# Auto-select best available device
device = DeviceConfig(device=DeviceType.AUTO)

# Explicit CUDA with specific GPU
device = DeviceConfig(device=DeviceType.CUDA, gpu_id=0)

# Enable mixed precision
device = DeviceConfig(
    device=DeviceType.CUDA,
    mixed_precision=True
)
```
