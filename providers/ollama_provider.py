"""Ollama provider for local LLM models."""

from typing import Optional
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseProvider, LLMResponse
import config
from config import OLLAMA_MODELS


class OllamaProvider(BaseProvider):
    """Ollama local LLM provider."""

    name = "ollama"

    def __init__(self):
        self.default_model = OLLAMA_MODELS["llama"]
        self._available = None

    @property
    def host(self):
        """Get host from current settings (supports hot reload)."""
        return config.settings.ollama_host

    def is_available(self) -> bool:
        """Check if Ollama is running locally."""
        if self._available is not None:
            return self._available

        try:
            client = ollama.Client(host=self.host)
            client.list()
            self._available = True
        except Exception:
            self._available = False

        return self._available

    def _get_model_id(self, model: Optional[str]) -> str:
        """Resolve model name to Ollama model ID."""
        if model is None:
            return self.default_model
        if model.lower() in OLLAMA_MODELS:
            return OLLAMA_MODELS[model.lower()]
        return model

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate using local Ollama model."""
        if not self.is_available():
            return LLMResponse(
                content="",
                model=model or "unknown",
                provider=self.name,
                error=f"Ollama not available at {self.host}",
            )

        model_id = self._get_model_id(model)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            import asyncio
            client = ollama.Client(host=self.host)

            # Run synchronous Ollama in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat(
                    model=model_id,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                    },
                )
            )

            content = response["message"]["content"]

            return LLMResponse(
                content=content,
                model=model_id,
                provider=self.name,
                tokens_used=response.get("eval_count"),
                finish_reason="stop",
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=model_id,
                provider=self.name,
                error=str(e),
            )
