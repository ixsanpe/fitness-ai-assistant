"""
Main entry point for the Fitness AI Assistant.

Provides a simple CLI interface to run common pipeline tasks.
"""

import argparse


def run_feature_pipeline(args):
    """Run feature pipeline (dataset building or embedding generation)."""
    from src.feature_pipeline import build_dataset, compute_embeddings

    if args.step == "build":
        build_dataset.main(["--config", args.config])
    elif args.step == "embed":
        compute_embeddings.main(["--config", args.config])


def run_training_pipeline(args):
    """Run MLflow experiments."""
    from src.training_pipeline import mlflow_experiments

    cmd_args = [
        "--mode",
        args.mode,
        "--configs",
        *args.configs,
        "--tracking_uri",
        args.tracking_uri,
    ]
    if args.max_samples:
        cmd_args.extend(["--max_samples", str(args.max_samples)])

    mlflow_experiments.main(argparse.Namespace(**vars(args)))


def run_inference_pipeline(args):
    """Run inference pipeline (vector DB creation or query interface)."""
    from src.inference_pipeline import create_db, demo_query, gradio_app

    if args.step == "create-db":
        create_db.main(["--config", args.config])
    elif args.step == "demo":
        demo_query.main(["--config", args.config, "--query", args.query])
    elif args.step == "app":
        gradio_app.main(["--config", args.config])


def main():
    parser = argparse.ArgumentParser(
        description="Fitness AI Assistant - Multi-modal exercise search and recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build dataset and generate embeddings
  python main.py feature build --config configs/feature_sentence.yaml
  python main.py feature embed --config configs/feature_clip.yaml

  # Run MLflow experiments
  python main.py train --mode compare --configs configs/feature_sentence.yaml configs/feature_clip.yaml

  # Create vector database and query
  python main.py inference create-db --config configs/inference.yaml
  python main.py inference demo --config configs/inference.yaml --query "shoulder exercises"
  python main.py inference app --config configs/inference.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="pipeline", help="Pipeline to run", required=True)

    # Feature pipeline
    feature_parser = subparsers.add_parser("feature", help="Run feature pipeline")
    feature_parser.add_argument("step", choices=["build", "embed"], help="Pipeline step")
    feature_parser.add_argument(
        "--config", default="configs/feature_sentence.yaml", help="Config file path"
    )

    # Training pipeline
    train_parser = subparsers.add_parser("train", help="Run training/experiments pipeline")
    train_parser.add_argument(
        "--mode", choices=["single", "compare"], default="compare", help="Experiment mode"
    )
    train_parser.add_argument(
        "--configs", nargs="+", default=["configs/feature_sentence.yaml"], help="Config files"
    )
    train_parser.add_argument(
        "--tracking_uri", default="sqlite:///mlruns/mlflow.db", help="MLflow tracking URI"
    )
    train_parser.add_argument("--max_samples", type=int, help="Limit samples for testing")

    # Inference pipeline
    inference_parser = subparsers.add_parser("inference", help="Run inference pipeline")
    inference_parser.add_argument(
        "step", choices=["create-db", "demo", "app"], help="Pipeline step"
    )
    inference_parser.add_argument(
        "--config", default="configs/inference.yaml", help="Config file path"
    )
    inference_parser.add_argument("--query", help="Query string (for demo)")

    args = parser.parse_args()

    # Route to appropriate pipeline
    if args.pipeline == "feature":
        run_feature_pipeline(args)
    elif args.pipeline == "train":
        run_training_pipeline(args)
    elif args.pipeline == "inference":
        run_inference_pipeline(args)


if __name__ == "__main__":
    main()
