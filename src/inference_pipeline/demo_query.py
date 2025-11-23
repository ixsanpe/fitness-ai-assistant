import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import load_config
from src.inference_pipeline.pipeline import InferencePipeline


def demo(query: str = "sit up", top_k: int = 5, config_path: str = None):
    """Run a demo query against the inference pipeline.

    Args:
        query: Query text
        top_k: Number of results to return
        config_path: Path to config file
    """
    print(f"Running demo query: '{query}' (top_k={top_k})")

    # Load configuration
    config = load_config("inference", config_path)
    print(f"Using config: {config.name}")
    print(f"Backend: {config.vector_db.backend}")

    try:
        pipe = InferencePipeline(config)
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        import traceback

        traceback.print_exc()
        return

    try:
        res = pipe.query(query, top_k=top_k)
        print(f"\n🔍 Found {len(res)} results:\n")

        for i, r in enumerate(res, 1):
            print(f"{i}. [{r.get('id', 'N/A')}] (score={r.get('score', 0):.4f})")
            if r.get("combined_text"):
                text = r["combined_text"][:200].replace("\n", " ")
                print(f"   {text}...")
            print()
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Demo query against inference pipeline")
    p.add_argument("--query", type=str, default="sit up", help="Query text")
    p.add_argument("--top_k", type=int, default=5, help="Number of results")
    p.add_argument(
        "--config", type=str, default="configs/inference.yaml", help="Path to config file"
    )
    args = p.parse_args()

    demo(query=args.query, top_k=args.top_k, config_path=args.config)

# Usage examples:
# python src/inference_pipeline/demo_query.py --query "sit up" --top_k 5
# python src/inference_pipeline/demo_query.py --query "push up" --config configs/inference.yaml
# python src/inference_pipeline/demo_query.py --query "bicep curl" --config configs/inference_prod.yaml --top_k 10
