"""ARAG - Agentic Retrieval-Augmented Generation Framework."""

__version__ = "0.1.0"

from evals.baselines.arag.upstream.core.config import Config
from evals.baselines.arag.upstream.core.context import AgentContext
from evals.baselines.arag.upstream.core.llm import LLMClient
from evals.baselines.arag.upstream.agent.base import BaseAgent
from evals.baselines.arag.upstream.tools.base import BaseTool
from evals.baselines.arag.upstream.tools.registry import ToolRegistry

__all__ = [
    "Config",
    "AgentContext", 
    "LLMClient",
    "BaseAgent",
    "BaseTool",
    "ToolRegistry",
    "__version__",
]
