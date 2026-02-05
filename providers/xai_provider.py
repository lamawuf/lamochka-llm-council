"""xAI (Grok) API provider."""

import httpx
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseProvider, LLMResponse
import config
from config import DIRECT_MODELS


class XAIProvider(BaseProvider):
    """xAI (Grok) API provider."""

    name = "xai"
    base_url = "https://api.x.ai/v1"

    def __init__(self):
        self.default_model = DIRECT_MODELS["grok"]

    @property
    def api_key(self):
        """Get API key from current settings (supports hot reload)."""
        return config.settings.xai_api_key

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
        """Generate using xAI Grok API."""
        if not self.is_available():
            return LLMResponse(
                content="",
                model=model or "unknown",
                provider=self.name,
                error="xAI API key not configured",
            )

        model_id = model or self.default_model
        if model_id == "grok":
            model_id = DIRECT_MODELS["grok"]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=config.settings.request_timeout) as client:
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
