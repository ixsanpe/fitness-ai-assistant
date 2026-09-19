"""Generation (LLM) configuration schema."""

from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """Config for the RAG generation step, backed by a local Ollama model."""

    enabled: bool = Field(default=False, description="Run a generation step over retrieved results")
    model_name: str = Field(default="qwen2.5:7b", description="Ollama model tag")
    base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    max_context_chars: int = Field(
        default=400, description="Chars of retrieved text per result included in the prompt"
    )
    timeout: float = Field(default=60.0, description="Ollama request timeout in seconds")
