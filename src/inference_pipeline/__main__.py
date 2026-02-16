"""CLI entry point for inference pipeline.

Usage:
    python -m src.inference_pipeline --query "push up" --top_k 5
    python -m src.inference_pipeline "sit up" --config configs/inference.yaml
"""
import argparse

from src.config import InferenceConfig, load_config
from src.inference_pipeline.pipeline import InferencePipeline


def main():
    """Main CLI entry point for inference pipeline."""
    parser = argparse.ArgumentParser(description="Query the inference pipeline")
    parser.add_argument("--query", nargs="+", help="Query text")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to config file")
    args = parser.parse_args()

    query_text = " ".join(args.query)

    try:
        config = load_config("inference", args.config)
        assert isinstance(config, InferenceConfig), "Loaded config is not an InferenceConfig instance"
        print(f"✅ Loaded config: {config.name}")

        pipeline = InferencePipeline(config)
        results = pipeline.query(query_text, top_k=args.top_k)

        print(f"\n🔍 Results for query: '{query_text}'\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. [{r['id']}] (score={r['score']:.4f})")
            if r.get("combined_text"):
                text_preview = r["combined_text"][:200].replace("\n", " ")
                print(f"   {text_preview}...\n")

        pipeline.close()
        print("✅ Pipeline closed successfully\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
