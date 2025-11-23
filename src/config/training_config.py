"""Training pipeline configuration for model training and fine-tuning."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .base import BaseConfig


class TrainingStrategy(str, Enum):
    """Training strategies."""

    SUPERVISED = "supervised"
    CONTRASTIVE = "contrastive"
    TRIPLET = "triplet"
    DISTILLATION = "distillation"
    FINE_TUNE = "fine_tune"


class LossFunction(str, Enum):
    """Loss functions for training."""

    CROSS_ENTROPY = "cross_entropy"
    MSE = "mse"
    COSINE_SIMILARITY = "cosine_similarity"
    TRIPLET_LOSS = "triplet_loss"
    CONTRASTIVE_LOSS = "contrastive_loss"
    MULTI_SIMILARITY_LOSS = "multi_similarity_loss"


class OptimizerType(str, Enum):
    """Optimizer types."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    RMSPROP = "rmsprop"


class SchedulerType(str, Enum):
    """Learning rate scheduler types."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class OptimizerConfig(BaseModel):
    """Configuration for optimizer."""

    optimizer_type: OptimizerType = Field(
        default=OptimizerType.ADAMW, description="Type of optimizer to use"
    )
    learning_rate: float = Field(default=2e-5, description="Initial learning rate", gt=0.0, le=1.0)
    weight_decay: float = Field(
        default=0.01, description="Weight decay (L2 regularization)", ge=0.0
    )

    # Optimizer-specific parameters
    betas: tuple[float, float] = Field(
        default=(0.9, 0.999), description="Beta parameters for Adam/AdamW"
    )
    momentum: float = Field(default=0.9, description="Momentum for SGD", ge=0.0, le=1.0)
    eps: float = Field(default=1e-8, description="Epsilon for numerical stability", gt=0.0)

    # Gradient clipping
    max_grad_norm: float | None = Field(
        default=1.0, description="Maximum gradient norm for clipping (None = no clipping)"
    )


class SchedulerConfig(BaseModel):
    """Configuration for learning rate scheduler."""

    scheduler_type: SchedulerType = Field(
        default=SchedulerType.LINEAR, description="Type of learning rate scheduler"
    )
    num_warmup_steps: int = Field(default=0, description="Number of warmup steps", ge=0)
    num_training_steps: int | None = Field(
        default=None, description="Total number of training steps (auto-calculated if None)"
    )

    # Scheduler-specific parameters
    num_cycles: int = Field(default=1, description="Number of cycles for cosine scheduler", ge=1)
    power: float = Field(default=1.0, description="Power for polynomial scheduler", gt=0.0)
    gamma: float = Field(
        default=0.1,
        description="Multiplicative factor for exponential/plateau scheduler",
        gt=0.0,
        le=1.0,
    )
    patience: int = Field(default=10, description="Patience for reduce on plateau scheduler", ge=1)


class DataConfig(BaseModel):
    """Configuration for training data."""

    train_path: Path = Field(
        default=Path("data/processed/train.jsonl"), description="Path to training data"
    )
    val_path: Path | None = Field(
        default=None, description="Path to validation data (None = use split from train)"
    )
    test_path: Path | None = Field(default=None, description="Path to test data")

    # Data splitting
    val_split: float = Field(
        default=0.1, description="Validation split ratio if val_path is None", ge=0.0, le=0.5
    )
    test_split: float = Field(
        default=0.1, description="Test split ratio if test_path is None", ge=0.0, le=0.5
    )

    # Data loading
    batch_size: int = Field(default=32, description="Training batch size", ge=1)
    eval_batch_size: int | None = Field(
        default=None, description="Evaluation batch size (None = same as batch_size)"
    )
    num_workers: int = Field(default=4, description="Number of data loading workers", ge=0)
    shuffle: bool = Field(default=True, description="Shuffle training data")
    drop_last: bool = Field(default=False, description="Drop last incomplete batch")

    # Data augmentation
    augmentation_prob: float = Field(
        default=0.0, description="Probability of applying augmentation", ge=0.0, le=1.0
    )


class CheckpointConfig(BaseModel):
    """Configuration for model checkpointing."""

    save_dir: Path = Field(
        default=Path("models/checkpoints"), description="Directory to save checkpoints"
    )
    save_strategy: str = Field(
        default="epoch",
        description="When to save checkpoints (epoch, steps, best)",
        pattern="^(epoch|steps|best)$",
    )
    save_steps: int = Field(
        default=500, description="Save checkpoint every N steps (if save_strategy='steps')", ge=1
    )
    save_total_limit: int | None = Field(
        default=3, description="Maximum number of checkpoints to keep (None = unlimited)"
    )
    load_best_model_at_end: bool = Field(
        default=True, description="Load best model at end of training"
    )

    # Model saving format
    save_format: str = Field(
        default="safetensors",
        description="Format to save model weights",
        pattern="^(safetensors|pytorch|onnx)$",
    )


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    eval_strategy: str = Field(
        default="epoch",
        description="When to evaluate (epoch, steps, no)",
        pattern="^(epoch|steps|no)$",
    )
    eval_steps: int = Field(
        default=500, description="Evaluate every N steps (if eval_strategy='steps')", ge=1
    )

    # Metrics
    metrics: list[str] = Field(
        default=["accuracy", "precision", "recall", "f1"],
        description="Metrics to compute during evaluation",
    )
    metric_for_best_model: str = Field(
        default="f1", description="Metric to use for selecting best model"
    )
    greater_is_better: bool = Field(
        default=True, description="Whether higher metric value is better"
    )


class TrainingConfig(BaseConfig):
    """Complete configuration for the training pipeline."""

    # Model
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Base model to train/fine-tune",
    )
    embedding_type: str = Field(default="sentence", description="Type of model architecture")

    # Training strategy
    strategy: TrainingStrategy = Field(
        default=TrainingStrategy.FINE_TUNE, description="Training strategy"
    )
    loss_function: LossFunction = Field(
        default=LossFunction.COSINE_SIMILARITY, description="Loss function to use"
    )

    # Training parameters
    num_epochs: int = Field(default=3, description="Number of training epochs", ge=1)
    max_steps: int | None = Field(
        default=None, description="Maximum number of training steps (overrides num_epochs)"
    )
    gradient_accumulation_steps: int = Field(
        default=1, description="Number of steps to accumulate gradients", ge=1
    )

    # Components
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig, description="Optimizer configuration"
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig, description="Learning rate scheduler configuration"
    )
    data: DataConfig = Field(default_factory=DataConfig, description="Data configuration")
    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig, description="Checkpoint configuration"
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation configuration"
    )

    # Regularization
    dropout: float = Field(default=0.1, description="Dropout probability", ge=0.0, le=0.9)
    label_smoothing: float = Field(
        default=0.0, description="Label smoothing factor", ge=0.0, le=0.5
    )

    # Advanced settings
    fp16: bool = Field(default=False, description="Use mixed precision training (FP16)")
    bf16: bool = Field(default=False, description="Use bfloat16 precision")
    use_cpu: bool = Field(default=False, description="Force CPU training")
    dataloader_pin_memory: bool = Field(default=True, description="Pin memory in dataloader")

    # Logging and monitoring
    logging_steps: int = Field(default=10, description="Log metrics every N steps", ge=1)
    log_level: str = Field(default="info", description="Logging level")
    report_to: list[str] = Field(
        default=["tensorboard"], description="Tools to report metrics to (tensorboard, wandb, etc.)"
    )

    # Experiment tracking
    run_name: str | None = Field(default=None, description="Name for this training run")
    tags: list[str] = Field(default_factory=list, description="Tags for experiment tracking")
    notes: str | None = Field(default=None, description="Notes about this training run")

    @field_validator("fp16", "bf16")
    @classmethod
    def validate_precision(cls, v, info):
        """Ensure only one precision mode is enabled."""
        if info.field_name == "bf16" and v:
            if info.data.get("fp16", False):
                raise ValueError("Cannot enable both fp16 and bf16")
        return v
