from src.llm.base import LLMProvider
from src.pipeline.config import LLMConfig
from src.utils.errors import ConfigError


def get_provider(cfg: LLMConfig) -> LLMProvider:
    if cfg.provider == "openai":
        from src.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(model=cfg.model, temperature=cfg.temperature)
    raise ConfigError(f"Unknown LLM provider '{cfg.provider}'. Available: openai")