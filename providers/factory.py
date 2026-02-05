"""Provider factory for creating LLM provider instances."""

from typing import Dict, Optional, Type

from .base import BaseProvider
from .openrouter import OpenRouterProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider
from .xai_provider import XAIProvider
from .ollama_provider import OllamaProvider
from config import settings


class ProviderFactory:
    """Factory for creating and managing LLM providers."""

    _providers: Dict[str, Type[BaseProvider]] = {
        "openrouter": OpenRouterProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "xai": XAIProvider,
        "ollama": OllamaProvider,
    }

    _instances: Dict[str, BaseProvider] = {}

    @classmethod
    def get(cls, provider_name: Optional[str] = None) -> BaseProvider:
        """Get a provider instance.

        Args:
            provider_name: Name of the provider. Uses default if not specified.

        Returns:
            Provider instance

        Raises:
            ValueError: If provider is not found
        """
        name = provider_name or settings.default_provider

        if name not in cls._providers:
            raise ValueError(
                f"Unknown provider: {name}. "
                f"Available: {list(cls._providers.keys())}"
            )

        # Return cached instance
        if name not in cls._instances:
            cls._instances[name] = cls._providers[name]()

        return cls._instances[name]

    @classmethod
    def get_available(cls) -> Dict[str, BaseProvider]:
        """Get all available (configured) providers."""
        available = {}
        for name in cls._providers:
            provider = cls.get(name)
            if provider.is_available():
                available[name] = provider
        return available

    @classmethod
    def get_best_available(cls) -> Optional[BaseProvider]:
        """Get the best available provider (prioritizes OpenRouter)."""
        priority = ["openrouter", "anthropic", "openai", "google", "xai", "ollama"]

        for name in priority:
            provider = cls.get(name)
            if provider.is_available():
                return provider

        return None

    @classmethod
    def model_to_provider(cls, model: str) -> str:
        """Determine which provider to use for a model shorthand."""
        model_provider_map = {
            "claude": "anthropic",
            "gpt4": "openai",
            "gemini": "google",
            "grok": "xai",
            "llama": "ollama",
            "mistral": "ollama",
            "mixtral": "ollama",
        }

        # If OpenRouter is available, use it for everything except local models
        openrouter = cls.get("openrouter")
        if openrouter.is_available() and model.lower() not in ["llama", "mistral", "mixtral"]:
            return "openrouter"

        return model_provider_map.get(model.lower(), settings.default_provider)


def get_provider(name: Optional[str] = None) -> BaseProvider:
    """Convenience function to get a provider."""
    return ProviderFactory.get(name)
