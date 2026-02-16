"""CLI entry point for feature pipeline.

Usage:
    python -m src.feature_pipeline --config configs/feature_sentence.yaml --sample_limit 100
"""

import argparse

from src.feature_pipeline.pipeline import run_pipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run the full feature pipeline")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--input_dir", type=str, help="Override input directory for raw data")
    parser.add_argument("--dataset_path", type=str, help="Override dataset path (if already built)")
    parser.add_argument("--sample_limit", type=int, help="Limit number of items to process")
    args = parser.parse_args()

    result = run_pipeline(
        config_path=args.config,
        input_dir=args.input_dir,
        dataset_path=args.dataset_path,
        sample_limit=args.sample_limit,
    )

    print("\n✅ Pipeline completed successfully!")
    print(f"Processed {result['num_items']} items")
    print(f"Embeddings shape: {result['embedding_shape']}")
    print(f"Device used: {result['device']}\n")


if __name__ == "__main__":
    main()
