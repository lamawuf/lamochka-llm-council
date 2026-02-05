"""Anthropic (Claude) direct API provider."""

from typing import Optional
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseProvider, LLMResponse
from config import settings, DIRECT_MODELS


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""

    name = "anthropic"

    def __init__(self):
        self.api_key = settings.anthropic_api_key
        self.client = AsyncAnthropic(api_key=self.api_key) if self.api_key else None
        self.default_model = DIRECT_MODELS["claude"]

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
        """Generate using Anthropic API."""
        if not self.is_available():
            return LLMResponse(
                content="",
                model=model or "unknown",
                provider=self.name,
                error="Anthropic API key not configured",
            )

        model_id = model or self.default_model
        if model_id == "claude":
            model_id = DIRECT_MODELS["claude"]

        try:
            response = await self.client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )

            content = response.content[0].text
            usage = response.usage

            return LLMResponse(
                content=content,
                model=model_id,
                provider=self.name,
                tokens_used=usage.input_tokens + usage.output_tokens if usage else None,
                finish_reason=response.stop_reason,
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=model_id,
                provider=self.name,
                error=str(e),
            )
