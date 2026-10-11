from src.llm.base import LLMProvider
from src.pipeline.config import LLMConfig
from src.utils.errors import ConfigError


def get_provider(cfg: LLMConfig) -> LLMProvider:
    if cfg.provider == "openai":
        from src.llm.openai_provider import OpenAIProvider

        return OpenAIProvider(
            model=cfg.model,
            temperature=cfg.temperature,
            base_url=cfg.base_url,
            api_key_env=cfg.api_key_env,
            requests_per_minute=cfg.requests_per_minute,
        )
    raise ConfigError(f"Unknown LLM provider '{cfg.provider}'. Available: openai")