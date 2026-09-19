"""Training pipeline configuration for model training and fine-tuning.

Deliberately minimal — only fields `src/training_pipeline/contrastive_finetune.py`
actually reads. `SentenceTransformerTrainingArguments` exposes many more knobs
(LR schedules, precision, checkpointing strategy, ...); add them here only once
a training script actually consumes them, not speculatively ahead of need.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from .base import BaseConfig


class LossFunction(StrEnum):
    """Loss functions for training.

    Only MULTIPLE_NEGATIVES_RANKING is implemented today. Add a value here
    alongside the training script that implements it.
    """

    MULTIPLE_NEGATIVES_RANKING = "multiple_negatives_ranking"


class OptimizerConfig(BaseModel):
    """Configuration for optimizer."""

    learning_rate: float = Field(default=2e-5, description="Initial learning rate", gt=0.0, le=1.0)
    weight_decay: float = Field(
        default=0.01, description="Weight decay (L2 regularization)", ge=0.0
    )


class SchedulerConfig(BaseModel):
    """Configuration for learning rate scheduler."""

    num_warmup_steps: int = Field(default=0, description="Number of warmup steps", ge=0)


class DataConfig(BaseModel):
    """Configuration for training data."""

    train_path: Path = Field(
        default=Path("data/training/contrastive_pairs.jsonl"), description="Path to training data"
    )
    val_split: float = Field(default=0.1, description="Validation split ratio", ge=0.0, le=0.5)
    batch_size: int = Field(default=32, description="Training batch size", ge=1)
    eval_batch_size: int | None = Field(
        default=None, description="Evaluation batch size (None = same as batch_size)"
    )
    drop_last: bool = Field(default=False, description="Drop last incomplete batch")


class CheckpointConfig(BaseModel):
    """Configuration for model checkpointing."""

    save_dir: Path = Field(
        default=Path("models/checkpoints"), description="Directory to save the trained model"
    )


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    eval_strategy: str = Field(
        default="epoch",
        description="When to evaluate (epoch, steps, no)",
        pattern="^(epoch|steps|no)$",
    )


class TrainingConfig(BaseConfig):
    """Complete configuration for the training pipeline."""

    seed: int = Field(default=42, description="Random seed for reproducibility")

    # Model
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Base model to train/fine-tune",
    )

    loss_function: LossFunction = Field(
        default=LossFunction.MULTIPLE_NEGATIVES_RANKING, description="Loss function to use"
    )

    # Training parameters
    num_epochs: int = Field(default=3, description="Number of training epochs", ge=1)

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

    # Logging and monitoring
    logging_steps: int = Field(default=10, description="Log metrics every N steps", ge=1)

    # Experiment tracking
    run_name: str | None = Field(default=None, description="Name for this training run")
