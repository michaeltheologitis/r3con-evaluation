from dotenv import load_dotenv

from evals.settings import settings

# Load API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...) from .env into os.environ
# so LiteLLM picks them up. Safe no-op if .env is missing.
load_dotenv(settings.ENV_FILE)

from .chat import litellm_chat_completion, litellm_chat_completion_full

__all__ = ["litellm_chat_completion", "litellm_chat_completion_full"]
