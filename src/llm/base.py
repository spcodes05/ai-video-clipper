from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Provider-agnostic interface. Add a local LLM later by subclassing this."""

    @abstractmethod
    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw model reply, which should be a JSON string."""