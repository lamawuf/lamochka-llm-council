"""Tests for LLM providers."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from providers import (
    ProviderFactory,
    OpenRouterProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    XAIProvider,
    OllamaProvider,
)
from providers.base import LLMResponse


class TestProviderFactory:
    """Test cases for ProviderFactory."""

    def test_get_provider_by_name(self):
        """Test getting provider by name."""
        provider = ProviderFactory.get("openrouter")
        assert isinstance(provider, OpenRouterProvider)

    def test_get_unknown_provider(self):
        """Test error on unknown provider."""
        with pytest.raises(ValueError) as exc_info:
            ProviderFactory.get("unknown_provider")
        assert "Unknown provider" in str(exc_info.value)

    def test_model_to_provider_mapping(self):
        """Test model to provider mapping."""
        assert ProviderFactory.model_to_provider("claude") in ["anthropic", "openrouter"]
        assert ProviderFactory.model_to_provider("gpt4") in ["openai", "openrouter"]
        assert ProviderFactory.model_to_provider("llama") == "ollama"


class TestLLMResponse:
    """Test cases for LLMResponse."""

    def test_successful_response(self):
        """Test successful response."""
        response = LLMResponse(
            content="Test content",
            model="gpt-4",
            provider="openai",
            tokens_used=100,
        )
        assert response.success is True
        assert response.content == "Test content"

    def test_error_response(self):
        """Test error response."""
        response = LLMResponse(
            content="",
            model="gpt-4",
            provider="openai",
            error="API Error",
        )
        assert response.success is False


class TestOpenRouterProvider:
    """Test cases for OpenRouter provider."""

    def test_availability_without_key(self):
        """Test provider is unavailable without API key."""
        with patch("config.settings.openrouter_api_key", None):
            provider = OpenRouterProvider()
            provider.api_key = None
            assert provider.is_available() is False

    def test_model_resolution(self):
        """Test model name resolution."""
        provider = OpenRouterProvider()
        model_id = provider._get_model_id("claude")
        assert "anthropic" in model_id or "claude" in model_id

    @pytest.mark.asyncio
    async def test_generate_without_key(self):
        """Test generate returns error without API key."""
        provider = OpenRouterProvider()
        provider.api_key = None

        response = await provider.generate("test prompt")
        assert response.success is False
        assert "not configured" in response.error


class TestOllamaProvider:
    """Test cases for Ollama provider."""

    def test_model_resolution(self):
        """Test Ollama model resolution."""
        provider = OllamaProvider()
        provider._available = True  # Mock availability

        model_id = provider._get_model_id("llama")
        assert "llama" in model_id.lower()

    def test_unavailable_when_not_running(self):
        """Test provider reports unavailable when Ollama not running."""
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.list.side_effect = Exception("Connection refused")
            provider = OllamaProvider()
            provider._available = None  # Reset cache
            assert provider.is_available() is False


class TestProviderIntegration:
    """Integration tests for providers (require API keys)."""

    @pytest.mark.skip(reason="Requires API key")
    @pytest.mark.asyncio
    async def test_openrouter_real_request(self):
        """Test real OpenRouter request."""
        provider = OpenRouterProvider()
        if not provider.is_available():
            pytest.skip("OpenRouter not configured")

        response = await provider.generate(
            prompt="Say 'test' and nothing else.",
            max_tokens=10,
        )
        assert response.success is True
        assert len(response.content) > 0
