# RAG next steps

## Next steps, roughly by effort

1. **Reranking** — a cross-encoder pass (e.g. a small `sentence-transformers`
   cross-encoder) over the top-k before they go to the generator. Natural next
   step once retrieval returns more than a handful of plausible candidates.
2. **FAISS-based ANN sweep** — `src/eval/index_sweep.py` currently sweeps
   FLAT/IVF_FLAT/HNSW through Milvus Lite, which rejects an explicit
   `index_type="HNSW"` in local mode (`AUTOINDEX` is the workaround — see the
   README). A FAISS sweep has no such restriction and is worth building as a
   cross-check, and doubles as the ANN/access-method practice this project
   was using to mirror the legal-project work.
3. **Try more embedding models** — `all-MiniLM-L6-v2` is fast but dated; try a
   current small model (Nomic Embed v2, BGE-base, E5-base) through
   `src/eval/benchmark_retrieval.py` and compare against the numbers already
   there.
4. **Wire Lance in for real** (currently just a demo) — see the "what
   integrating Lance for real would take" section in `docs/milvus_vs_lance.md`
   for the concrete steps.
5. **Iterative/agentic retrieval** (SID-1-style) — worth knowing about, not
   worth building at this dataset size (873 exercises, exact search is already
   free). Revisit if the dataset grows enough that single-shot dense retrieval
   starts missing things a re-query would catch. See
   `agentic_search_ladder.md` for a concrete, scoped path from the current
   single-shot `answer()` up to tool-calling + interleaved search.

## Datasets for testing these concepts (not fitness-specific)

The fitness dataset (100-873 rows, one row per exercise) is too small and too
uniform to stress-test hybrid search, reranking, or multi-hop agentic
retrieval — these need either built-in relevance judgments (so eval isn't
itself a project) or questions that genuinely require multiple retrieval
hops. Two picks, no scraping needed (both load via HuggingFace `datasets`):

- **HotpotQA** — ~113K questions over Wikipedia, purpose-built for multi-hop
  QA: each question requires combining facts from *two different* paragraphs,
  with gold "supporting facts" labels marking exactly which passages should
  be retrieved. Best fit for testing `agentic_search_ladder.md` rungs 3-4
  (interleaved reasoning + search, decompose + parallelize) — lets you
  directly measure whether an iterative search loop beats single-shot
  retrieval instead of eyeballing it. Ships as a "distractor" setting (10
  paragraphs/question, fast local iteration) and a "full-wiki" setting
  (~5M paragraphs, real scale). `datasets.load_dataset("hotpot_qa")`.
- **BEIR** — a benchmark suite (SciFact, FiQA, NFCorpus, TREC-COVID,
  MS MARCO, etc.) in one consistent format, every dataset shipping with
  qrels (query → relevant-doc judgments) so recall@k/NDCG/MRR are computed
  for you. This is the one for validating hybrid search and reranking
  empirically instead of by feel. Start with SciFact (~5K docs, minutes to
  index) to iterate fast; use MS MARCO (8.8M passages) once ANN indexing
  (IVF/HNSW vs. flat scan) needs to actually matter.

Rough split: HotpotQA for the agentic-ladder work, a small BEIR subset for
hybrid-search/reranking/eval work, MS MARCO later for real production scale.

## Comparing against your job's RAG stack

Useful axes to check when comparing Monday:

- **Retrieval**: dense-only, or hybrid (dense + BM25/sparse)? Fused how?
- **Reranking**: cross-encoder reranker present? At what stage?
- **Chunking**: how is source content split before embedding — fixed-size,
  semantic, per-document (this project embeds one row per exercise, no
  chunking at all)?
- **Iteration**: single retrieve-then-generate, or can the system re-query
  based on what it read (agentic RAG)?
- **Generation model**: local/open vs. hosted frontier API — and why that
  choice was made (cost, latency, data residency, quality bar).
- **Grounding/citations**: does generation cite which retrieved doc backs each
  claim, or just narrate over the context like the current `OllamaGenerator`
  prompt does?
- **Eval**: how is retrieval/generation quality measured — golden queries,
  human eval, LLM-as-judge, recall@k?
- **Scale**: row/document count, and whether that changes which of the above
  actually matter (most of this list is close to irrelevant at 100 rows, and
  increasingly mandatory past ~100K).
