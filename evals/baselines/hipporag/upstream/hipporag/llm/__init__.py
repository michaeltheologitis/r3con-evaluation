import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .openai_gpt import CacheOpenAI
from .base import BaseLLM
# HIPPORAG-CONNECTOR DEVIATION (D3): the BedrockLLM (boto3) and TransformersLLM
# (torch/transformers) backends are not imported — they pull heavy/optional deps
# we don't use (our connector injects a litellm-backed LLM via _get_llm_class,
# so these branches are never taken). The vendored files remain on disk,
# byte-for-byte, just unimported. See PROVENANCE.md.
# from .bedrock_llm import BedrockLLM
# from .transformers_llm import TransformersLLM


logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    if config.llm_base_url is not None and 'localhost' in config.llm_base_url and os.getenv('OPENAI_API_KEY') is None:
        os.environ['OPENAI_API_KEY'] = 'sk-'

    # DEVIATION (D3): bedrock/Transformers branches dropped with their imports.
    return CacheOpenAI.from_experiment_config(config)
    