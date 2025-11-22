from .config import InferenceConfig
from src.inference_pipeline.pipeline import InferencePipeline


def create_pipeline(cfg: InferenceConfig) -> InferencePipeline:
    """Create and return an InferencePipeline according to the provided config."""
    return InferencePipeline(
        model_name=cfg.model_name,
        embedding_type=cfg.embedding_type,
        device=cfg.device,
        backend=cfg.backend,
        milvus_host=cfg.milvus_host,
        milvus_port=cfg.milvus_port,
        milvus_collection=cfg.milvus_collection,
        milvus_vector_field=cfg.milvus_vector_field,
        milvus_output_fields=cfg.milvus_output_fields,
    )
