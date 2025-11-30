"""
Context managers for MLflow experiment tracking.

This module provides reusable context managers for common experiment patterns.
"""

import time
from typing import Any

import mlflow
import torch


class TimedExperiment:
    """Context manager for tracking experiment timing and resource usage.

    Automatically logs:
    - Total experiment duration
    - GPU memory usage (if available)
    - Exception information if experiment fails

    Example:
        >>> with TimedExperiment(run_name="my_experiment") as exp:
        ...     exp.log_stage("data_loading")
        ...     # Load data
        ...     exp.log_stage("training")
        ...     # Train model
        ...     exp.log_metric("accuracy", 0.95)
    """

    def __init__(self, run_name: str, tags: dict[str, str] | None = None):
        """Initialize timed experiment.

        Args:
            run_name: Name of the MLflow run
            tags: Optional tags to add to the run
        """
        self.run_name = run_name
        self.tags = tags or {}
        self.start_time = None
        self.stage_times = {}
        self.current_stage = None
        self.current_stage_start = None
        self.active_run = None

    def __enter__(self):
        """Start the MLflow run and timing."""
        self.start_time = time.time()
        self.active_run = mlflow.start_run(run_name=self.run_name, tags=self.tags)

        # Log system info
        mlflow.log_param("device", "cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            mlflow.log_param("gpu_name", torch.cuda.get_device_name(0))
            torch.cuda.reset_peak_memory_stats()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the run and log final metrics."""
        # Log final stage if any
        if self.current_stage:
            self._finalize_current_stage()

        # Log total duration
        total_duration = time.time() - self.start_time
        mlflow.log_metric("total_duration_seconds", total_duration)

        # Log GPU memory if used
        if torch.cuda.is_available():
            max_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            mlflow.log_metric("max_gpu_memory_mb", max_memory_mb)

        # Log exception info if failed
        if exc_type is not None:
            mlflow.log_param("error_type", exc_type.__name__)
            mlflow.log_param("error_message", str(exc_val))
            mlflow.set_tag("status", "FAILED")
        else:
            mlflow.set_tag("status", "SUCCESS")

        # End the MLflow run
        mlflow.end_run()

        # Don't suppress exceptions
        return False

    def log_stage(self, stage_name: str):
        """Start tracking a new stage of the experiment.

        Args:
            stage_name: Name of the stage (e.g., "data_loading", "training")
        """
        # Finalize previous stage if exists
        if self.current_stage:
            self._finalize_current_stage()

        # Start new stage
        self.current_stage = stage_name
        self.current_stage_start = time.time()
        print(f"📊 Stage: {stage_name}")

    def _finalize_current_stage(self):
        """Finalize the current stage and log its duration."""
        if self.current_stage and self.current_stage_start:
            duration = time.time() - self.current_stage_start
            self.stage_times[self.current_stage] = duration
            mlflow.log_metric(f"stage_{self.current_stage}_duration", duration)
            print(f"   ✓ {self.current_stage}: {duration:.2f}s")

    def log_metric(self, key: str, value: float, step: int | None = None):
        """Log a metric to MLflow.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number for time-series metrics
        """
        mlflow.log_metric(key, value, step=step)

    def log_param(self, key: str, value: Any):
        """Log a parameter to MLflow.

        Args:
            key: Parameter name
            value: Parameter value
        """
        mlflow.log_param(key, value)
