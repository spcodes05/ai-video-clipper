import logging
import os

from src.llm.base import LLMProvider
from src.utils.errors import LLMError

log = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, temperature: float = 0.2, api_key: str | None = None, timeout: float = 90.0):
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMError("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=key, timeout=timeout, max_retries=3)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Could not initialise the OpenAI client: {exc}") from exc
        self.model = model
        self.temperature = temperature

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return resp.choices[0].message.content or ""