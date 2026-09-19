"""Grounded answer generation over retrieved results, via a local Ollama model.

This is the "G" in RAG: it takes whatever InferencePipeline.query() already
retrieved and asks a local LLM (served by `ollama serve`) to answer the user's
question using only that context. Uses Ollama's REST API directly (stdlib
urllib, no extra dependency) rather than the `ollama` package.
"""

import json
import urllib.error
import urllib.request

SYSTEM_PROMPT = (
    "You are a fitness assistant. Answer the user's question using ONLY the "
    "exercises listed in the context below — do not invent exercises that "
    "aren't there. If none of them fit, say so. Keep the answer short and "
    "reference exercises by name."
)


class OllamaGenerator:
    """Generates a grounded answer from retrieved results using a local Ollama model."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _build_prompt(self, query: str, hits: list[dict], max_context_chars: int) -> str:
        lines = []
        for i, hit in enumerate(hits, 1):
            name = hit.get("name") or hit.get("id") or f"result_{i}"
            text = (hit.get("combined_text") or hit.get("text") or "")[:max_context_chars]
            lines.append(f"{i}. {name}: {text}" if text else f"{i}. {name}")
        context = "\n".join(lines) if lines else "(no results found)"
        return f"Context (retrieved exercises):\n{context}\n\nQuestion: {query}"

    def generate(self, query: str, hits: list[dict], max_context_chars: int = 400) -> str:
        """Call the local Ollama model and return its answer text.

        Raises:
            RuntimeError: if the Ollama server can't be reached (e.g. `ollama
                serve` isn't running) or returns an error response.
        """
        prompt = self._build_prompt(query, hits, max_context_chars)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url} — is `ollama serve` "
                f"running and is '{self.model_name}' pulled? ({e})"
            ) from e

        if "error" in body:
            raise RuntimeError(f"Ollama error: {body['error']}")

        return body["message"]["content"]
