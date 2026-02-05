"""Base provider interface for LLM APIs."""

from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.content is not None


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str = "base"

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt/query
            system_prompt: System instructions
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum response tokens

        Returns:
            LLMResponse with the generated content
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and available."""
        pass

    async def health_check(self) -> bool:
        """Test if the provider is working."""
        try:
            response = await self.generate(
                prompt="Say 'OK' and nothing else.",
                max_tokens=10,
            )
            return response.success
        except Exception:
            return False
