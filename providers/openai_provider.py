"""OpenAI direct API provider."""

from typing import Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseProvider, LLMResponse
import config
from config import DIRECT_MODELS


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    name = "openai"

    def __init__(self):
        self.default_model = DIRECT_MODELS["gpt4"]
        self._client = None

    @property
    def api_key(self):
        """Get API key from current settings (supports hot reload)."""
        return config.settings.openai_api_key

    @property
    def client(self):
        """Lazy client initialization with current API key."""
        if self._client is None and self.api_key:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        return bool(self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate using OpenAI API."""
        if not self.is_available():
            return LLMResponse(
                content="",
                model=model or "unknown",
                provider=self.name,
                error="OpenAI API key not configured",
            )

        model_id = model or self.default_model
        if model_id == "gpt4":
            model_id = DIRECT_MODELS["gpt4"]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = response.choices[0].message.content
            usage = response.usage

            return LLMResponse(
                content=content,
                model=model_id,
                provider=self.name,
                tokens_used=usage.total_tokens if usage else None,
                finish_reason=response.choices[0].finish_reason,
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=model_id,
                provider=self.name,
                error=str(e),
            )
