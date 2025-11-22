from src.inference_pipeline.pipeline import InferencePipeline


def demo(query: str = "sit up", top_k: int = 5):
    print(f"Running demo query: '{query}' (top_k={top_k})")
    try:
        pipe = InferencePipeline(backend="milvus")
    except Exception as e:
        print("Failed to initialize pipeline:", e)
        return
    try:
        res = pipe.query(query, top_k=top_k)
        for r in res:
            print(f"[{r['idx']}] {r['id']} (score={r['score']:.4f})")
            if r.get("combined_text"):
                print("  ", r["combined_text"][:300].replace('\n',' '))
            print()
    except Exception as e:
        print("Query failed:", e)


if __name__ == "__main__":
    demo()
