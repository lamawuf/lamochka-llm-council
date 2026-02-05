"""OpenRouter provider - unified access to multiple LLMs."""

import httpx
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseProvider, LLMResponse
from config import settings, OPENROUTER_MODELS


class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider."""

    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def __init__(self):
        self.api_key = settings.openrouter_api_key
        self.default_model = OPENROUTER_MODELS["claude"]

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_model_id(self, model: Optional[str]) -> str:
        """Resolve model name to OpenRouter model ID."""
        if model is None:
            return self.default_model
        # Check if it's a shorthand
        if model.lower() in OPENROUTER_MODELS:
            return OPENROUTER_MODELS[model.lower()]
        # Assume it's a full model ID
        return model

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
        """Generate using OpenRouter API."""
        if not self.is_available():
            return LLMResponse(
                content="",
                model=model or "unknown",
                provider=self.name,
                error="OpenRouter API key not configured",
            )

        model_id = self._get_model_id(model)
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/llamich/lamochka-llm-council",
            "X-Title": "Lamochka LLM Council",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})

                return LLMResponse(
                    content=content,
                    model=model_id,
                    provider=self.name,
                    tokens_used=usage.get("total_tokens"),
                    finish_reason=data["choices"][0].get("finish_reason"),
                )

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
            return LLMResponse(
                content="",
                model=model_id,
                provider=self.name,
                error=error_msg,
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=model_id,
                provider=self.name,
                error=str(e),
            )
