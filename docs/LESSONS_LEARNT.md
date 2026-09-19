## Lessons learnt in the project:
### Setup the project with `uv`
Create venv, pyproject.toml and setup.sh \
In the setup.sh we define the installation of the package based on backend for torch. \
It is useful to distinguish dev dependencies vs. prod.
### Structure the project
- FTI: Feature, Training Inference architecture
1. The _feature pipeline_ transforms your data into features & labels, which are stored and versioned in a feature store. The feature store will act as the central repository of your features.
2. The _training pipeline_ ingests a specific version of the features & labels from the feature store and outputs the trained model weights, which are stored and versioned inside a model registry.
3. The _inference pipeline_ uses a given version of the features from the feature store and downloads a specific version of the model from the model registry.
[Project idea: LLM twin](https://medium.com/decodingai/an-end-to-end-framework-for-production-ready-llm-systems-by-building-your-llm-twin-2cc6bb01141f)
[FTI pipelines](https://www.hopsworks.ai/post/mlops-to-ml-systems-with-fti-pipelines)

- The configuration should be stated outside `src` for: environment-specific, user-editable, not packaged.
### Build the dataset
JOSNLines files are easier to split and process in parallel (real-time, parallel processing)
The vector db comparison: https://docs.langchain.com/oss/python/integrations/vectorstores
### Multimodal embeddings: text and images
Ability to translate diverse data types into a common representational format in a high-dimensional space, where data of similar semantic are placed close together.
[Link info](https://milvus.io/docs/embeddings.md#Embedding-Overview)
#### CLIP
CLIP Processor: includes the image processor and the tokenizer. The inputs are set in TextKwargs

### CLIP embeddings vs text embeddings
- CLIP aligns text ↔ **images**, not text ↔ text. Bad at text-only retrieval (recall@10 0.033 vs 0.356 for a plain sentence embedder, on this project's eval set).
- Use CLIP only when images are actually involved (image search, image↔text). For text-only queries, use a text embedder instead.

### Fusion: dense vs hybrid (BM25 + dense, RRF)
- Dense alone barely beat BM25 here: short, keyword-heavy queries are exactly what lexical search is built for.
- Hybrid (BM25 + dense, fused with Reciprocal Rank Fusion) beat both alone on every metric.
- RRF fuses by **rank**, not raw score — needed because BM25 and cosine-similarity scores aren't on comparable scales.
- Fuse from a deeper candidate pool (e.g. top-20) per retriever, then trim to top-k — fusing already-truncated top-k lists starves RRF of signal.
- See `src/eval/fusion.py` (RRF) and `src/eval/benchmark_retrieval.py` (the comparison).

### BM25
- Pure lexical, no embeddings/training needed — cheap baseline.
- Strong on short, exact-term queries; weaker on paraphrase/semantic queries dense models handle better.
- Best used *alongside* dense (hybrid), not instead of it.
- See `src/inference_pipeline/backends/bm25.py`.

### Model evaluation
Most multimodal models use zero-shot learning (ZSL) where, at test time, a learner observes samples from classes which were not observed during training, and needs to predict the class that they belong to.

### mlflow for experiment tracking
- Wrap each run in `with mlflow.start_run(...)` — one run per config/model comparison.
- `log_param` for setup (model name, batch size, device, loss fn); `log_metric` for numeric results (encoding time, samples/sec, embedding dim, cosine-sim stats). Keeping that split consistent is what makes runs comparable in the UI table.
- No server needed to start: local `file:./mlruns` works out of the box. Point `--tracking_uri` at `sqlite:///...` or a remote server only when runs need to be shared/persisted beyond one machine.

### Contrastive fine-tuning (domain adaptation)
- `MultipleNegativesRankingLoss` = in-batch negatives: every other pair's positive in the batch doubles as this pair's negative. No manual negative mining needed.
- Fine-tuning gave a big jump here (recall@10 0.356 → 0.813) — but that's "learned this domain's vocabulary/phrasing," not proof it generalizes to unseen phrasing, since eval and training pairs were bootstrapped from the *same* metadata/templates.
- Log the before/after retrieval metrics into MLflow too (not just training loss), so the comparison is queryable later, not just printed once.

## Pending steps:
- logging instead of printing
- containers with docker
- build an mcp, connect to notion
- work with images: pose estimation

## Resources:
- https://bhavishyapandit9.substack.com/p/building-multimodal-embeddings-a
- https://huggingface.co/learn/cookbook/en/faiss_with_hf_datasets_and_clip
- https://towardsdatascience.com/multimodal-embeddings-an-introduction-5dc36975966f/
