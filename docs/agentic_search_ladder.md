# The agentic search ladder:

Seven maturity rungs the RAG/search field has climbed, `Pipeline → tool →
behavior → policy → system`, and where things we've discussed land on them.

| Rung | Stage | Layer | Where it sits |
|---|---|---|---|
| 1 | Retrieve-then-read RAG | Pipeline | **This project, exactly as built.** `InferencePipeline.answer()` is one embed → one search → one generate call, no loop back. |
| 2 | Search as a native tool · MCP | Tool | Search exposed as something an LLM can call itself, rather than a fixed pipeline step. **Search API** lives here — a tool any agent can invoke. |
| 3 | Interleaved reasoning + search | Behavior | ReAct-style: reason, search, reason about what came back, search again. |
| 4 | Decompose + parallelize | Behavior | Break a compound query into sub-queries, fan them out. |
| 5 | Adaptive: when *not* to search | Policy | Deciding retrieval isn't even needed for a given query — a real cost/latency lever. |
| 6 | RL-trained search behavior | Policy | **SID-1's rung.** Rungs 3-5 are normally hand-built scaffolding; SID-1 trains the decision of when/how to search as a learned policy via RL instead. |
| 7 | Deep research agents + context mgmt | System (2025-26) | Exa Search and OpenAI/Anthropic/Google "Deep Research" products — query expansion + reasoning + citation/context management packaged as a full system. |

See `rag_next_steps.md` for the broader RAG roadmap this fits into, and
`README.md` for the underlying concepts (hybrid search, reranking, eval).

## Climbing plan for this project

Rungs 6-7 need RL training infra or a whole product's worth of context/citation
management — not worth building at this project's scale (hundreds of rows,
one local model). Rungs 2-5, though, are a genuinely achievable, incremental
build on top of what already exists, using Ollama's tool-calling support
(`qwen2.5:7b` supports it) instead of the fixed `pipeline.answer()` call.

**Step 1 — Rung 2: search as a tool.**
Define `search_exercises(query, top_k)` as an Ollama tool schema wrapping
`InferencePipeline.query()`. Send it in the `tools` field of the `/api/chat`
call instead of pre-running retrieval ourselves. The model decides *whether*
and *how* to call it, instead of `answer()` always retrieving first.
This alone is most of the engineering lift — everything below is a loop
around this same tool call.

**Step 2 — Rung 3: interleave reasoning and search.**
Replace the current single request/response with a loop: send the
conversation, if the model responds with a tool call, run it, append the
result as a tool message, send again — repeat until the model returns a plain
answer (cap iterations, e.g. 3-4, to bound latency on a local 7B model).

**Step 3 — Rung 5: adaptive search comes free.**
Once search is a tool the model chooses to invoke (rung 2), "don't search for
things you already know" isn't separate work — it's the model simply not
calling the tool. Worth explicitly testing with a query that needs no
retrieval ("what's a good rep range for hypertrophy?") vs. one that does, to
confirm the policy behaves sensibly rather than always/never calling it.

**Step 4 — Rung 4: decompose + parallelize.**
For a compound query ("give me a push exercise and a pull exercise for a home
gym"), let the model issue multiple tool calls in one turn (Ollama supports
multiple tool calls per response) and run them — sequentially is fine given a
single local model instance; true parallelism only matters once search calls
are the bottleneck, which they aren't at this corpus size.

**Not planned:** rung 6 (RL-training a model to search) and rung 7 (a full
deep-research system with citation/context management across long sessions).
