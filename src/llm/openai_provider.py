import logging
import os
import re
import time

from src.llm.base import LLMProvider
from src.utils.errors import LLMError

log = logging.getLogger(__name__)

_RETRY_IN = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)
_MAX_RATE_LIMIT_WAIT = 120.0  # if a provider asks for longer (e.g. a daily quota), give up instead


class OpenAIProvider(LLMProvider):
    """Any OpenAI-compatible chat API (OpenAI, Gemini, Groq, OpenRouter, Ollama...)."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        api_key: str | None = None,
        timeout: float = 90.0,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        requests_per_minute: float | None = None,
        max_attempts: int = 6,
    ):
        key = api_key or os.getenv(api_key_env)
        if not key and base_url and any(h in base_url for h in ("localhost", "127.0.0.1")):
            key = "local"  # local servers such as Ollama need no real key
        if not key:
            raise LLMError(f"{api_key_env} is not set. Add it to your .env file (see .env.example).")
        try:
            from openai import OpenAI

            # Retries are handled below (so we can honor "retry in Ns" hints), not by the SDK.
            self._client = OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Could not initialise the OpenAI client: {exc}") from exc
        self.model = model
        self.temperature = temperature
        self.max_attempts = max(1, max_attempts)
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._last_call = float("-inf")

    def _throttle(self) -> None:
        """Keep at least min_interval seconds between requests (free tiers allow few per minute)."""
        if self.min_interval > 0:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                log.info("Pacing requests: waiting %.0fs", wait)
                time.sleep(wait)
        self._last_call = time.monotonic()

    @staticmethod
    def _retry_wait(exc: Exception, attempt: int) -> float | None:
        """Seconds to wait before retrying this error, or None if retrying is pointless."""
        text = str(exc)
        if "insufficient_quota" in text or "credit_balance_exhausted" in text:
            return None  # out of credits: waiting will not help
        status = getattr(exc, "status_code", None)
        if status == 429:
            match = _RETRY_IN.search(text)
            delay = float(match.group(1)) + 1.0 if match else 20.0 * attempt
            return delay if delay <= _MAX_RATE_LIMIT_WAIT else None
        if status is not None and status >= 500:
            return min(2.0 * 2**attempt, 30.0)
        if type(exc).__name__ in ("APIConnectionError", "APITimeoutError"):
            return min(2.0 * 2**attempt, 30.0)
        return None

    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        for attempt in range(1, self.max_attempts + 1):
            self._throttle()
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
                return resp.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                wait = self._retry_wait(exc, attempt)
                if wait is None or attempt >= self.max_attempts:
                    raise LLMError(f"LLM request failed: {exc}") from exc
                log.warning(
                    "LLM request failed (%s); retrying in %.0fs (attempt %d/%d)",
                    type(exc).__name__, wait, attempt, self.max_attempts,
                )
                time.sleep(wait)
        raise LLMError("LLM request failed")  # unreachable, keeps type-checkers happy