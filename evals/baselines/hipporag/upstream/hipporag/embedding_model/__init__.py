# HIPPORAG-CONNECTOR DEVIATION (D3): the local-embedder backends (Contriever /
# GritLM / NV-Embed-v2 / Cohere / Transformers / VLLM) are NOT imported at module
# top — they pull torch / transformers / gritlm / sentence_transformers, which we
# don't use (our connector injects an OpenAI-embedding seam via
# _get_embedding_model_class). The vendored files remain on disk byte-for-byte,
# just unimported. Only `base` (the ABC + config) stays, since embedding_store
# and our seam depend on it. See PROVENANCE.md.
from .base import EmbeddingConfig, BaseEmbeddingModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    # DEVIATION (D3): the heavy backends are lazy-imported per-branch so this
    # module imports without torch/transformers. In practice the connector
    # monkeypatches this function to return its OpenAI-embedding seam, so only the
    # text-embedding branch is ever reachable here.
    if "text-embedding" in embedding_model_name:
        from .OpenAI import OpenAIEmbeddingModel
        return OpenAIEmbeddingModel
    if "GritLM" in embedding_model_name:
        from .GritLM import GritLMEmbeddingModel
        return GritLMEmbeddingModel
    if "NV-Embed-v2" in embedding_model_name:
        from .NVEmbedV2 import NVEmbedV2EmbeddingModel
        return NVEmbedV2EmbeddingModel
    if "contriever" in embedding_model_name:
        from .Contriever import ContrieverModel
        return ContrieverModel
    if "cohere" in embedding_model_name:
        from .Cohere import CohereEmbeddingModel
        return CohereEmbeddingModel
    if embedding_model_name.startswith("Transformers/"):
        from .Transformers import TransformersEmbeddingModel
        return TransformersEmbeddingModel
    if embedding_model_name.startswith("VLLM/"):
        from .VLLM import VLLMEmbeddingModel
        return VLLMEmbeddingModel
    assert False, f"Unknown embedding model name: {embedding_model_name}"