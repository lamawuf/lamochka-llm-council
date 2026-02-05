"""Google (Gemini) direct API provider."""

from typing import Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseProvider, LLMResponse
from config import settings, DIRECT_MODELS


class GoogleProvider(BaseProvider):
    """Google Gemini API provider."""

    name = "google"

    def __init__(self):
        self.api_key = settings.google_api_key
        self.default_model = DIRECT_MODELS["gemini"]
        if self.api_key:
            genai.configure(api_key=self.api_key)

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
        """Generate using Google Gemini API."""
        if not self.is_available():
            return LLMResponse(
                content="",
                model=model or "unknown",
                provider=self.name,
                error="Google API key not configured",
            )

        model_id = model or self.default_model
        if model_id == "gemini":
            model_id = DIRECT_MODELS["gemini"]

        try:
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # Create model with system instruction
            model_instance = genai.GenerativeModel(
                model_name=model_id,
                system_instruction=system_prompt,
                generation_config=generation_config,
            )

            # Generate response (async wrapper)
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model_instance.generate_content(prompt)
            )

            content = response.text

            return LLMResponse(
                content=content,
                model=model_id,
                provider=self.name,
                tokens_used=None,  # Gemini doesn't expose token count easily
                finish_reason="stop",
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=model_id,
                provider=self.name,
                error=str(e),
            )
