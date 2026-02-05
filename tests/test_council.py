"""Tests for the council module."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from council import LLMCouncil
from schemas import CouncilConfig, RoleAssignment, StageResult
from providers.base import LLMResponse


class TestLLMCouncil:
    """Test cases for LLMCouncil class."""

    @pytest.fixture
    def mock_config(self):
        """Create a test configuration."""
        return CouncilConfig(
            roles=[
                RoleAssignment(role="chairman", model="claude"),
                RoleAssignment(role="critic", model="grok"),
                RoleAssignment(role="researcher", model="gemini"),
            ],
            max_debate_rounds=1,
            enable_web_search=False,
        )

    @pytest.fixture
    def council(self, mock_config):
        """Create a council instance."""
        return LLMCouncil(mock_config)

    def test_council_initialization(self, council, mock_config):
        """Test council initializes with correct config."""
        assert council.config == mock_config
        assert council.results == []
        assert council.total_tokens == 0

    def test_anonymize_responses(self, council):
        """Test response anonymization."""
        responses = {
            "Critic": "This is wrong because...",
            "Researcher": "The data shows...",
        }
        anonymized = council._anonymize_responses(responses)

        assert "Response A" in anonymized
        assert "Response B" in anonymized
        assert "Critic" not in anonymized
        assert "Researcher" not in anonymized
        assert "This is wrong because..." in anonymized

    def test_get_role_config(self, council):
        """Test role configuration retrieval."""
        critic = council._get_role_config("critic")
        assert critic is not None
        assert critic.name == "Critic"

        unknown = council._get_role_config("unknown_role")
        assert unknown is None

    @pytest.mark.asyncio
    async def test_generate_response_success(self, council):
        """Test successful response generation."""
        mock_response = LLMResponse(
            content="Test response",
            model="test-model",
            provider="test",
            tokens_used=100,
        )

        with patch.object(council, "_get_provider_for_role") as mock_provider:
            mock_provider.return_value.generate = AsyncMock(return_value=mock_response)

            assignment = RoleAssignment(role="critic", model="grok")
            content, tokens = await council._generate_response(assignment, "test prompt")

            assert content == "Test response"
            assert tokens == 100

    @pytest.mark.asyncio
    async def test_generate_response_error(self, council):
        """Test error handling in response generation."""
        mock_response = LLMResponse(
            content="",
            model="test-model",
            provider="test",
            error="API Error",
        )

        with patch.object(council, "_get_provider_for_role") as mock_provider:
            mock_provider.return_value.generate = AsyncMock(return_value=mock_response)

            assignment = RoleAssignment(role="critic", model="grok")
            content, tokens = await council._generate_response(assignment, "test prompt")

            assert "[Error:" in content
            assert tokens == 0


class TestStageResult:
    """Test cases for StageResult."""

    def test_stage_result_creation(self):
        """Test StageResult creation."""
        result = StageResult(
            stage=1,
            stage_name="Stage 1: First Opinions",
            responses={"Critic": "Response 1"},
        )

        assert result.stage == 1
        assert result.stage_name == "Stage 1: First Opinions"
        assert "Critic" in result.responses


class TestCouncilConfig:
    """Test cases for CouncilConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CouncilConfig()

        assert len(config.roles) == 4
        assert config.max_debate_rounds == 3
        assert config.disagreement_threshold == 7.0
        assert config.require_hitl is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = CouncilConfig(
            roles=[RoleAssignment(role="critic", model="grok")],
            max_debate_rounds=5,
            disagreement_threshold=8.0,
            require_hitl=True,
        )

        assert len(config.roles) == 1
        assert config.max_debate_rounds == 5
        assert config.disagreement_threshold == 8.0
        assert config.require_hitl is True

    def test_manual_mode_config(self):
        """Test manual mode configuration."""
        config = CouncilConfig(manual_mode=True)
        assert config.manual_mode is True


class TestDisagreementCheck:
    """Test cases for disagreement checking."""

    @pytest.mark.asyncio
    async def test_disagreement_parsing(self):
        """Test parsing of disagreement check response."""
        council = LLMCouncil()

        mock_response = LLMResponse(
            content='{"disagreement_score": 8, "key_conflicts": ["approach", "scope"]}',
            model="test",
            provider="test",
        )

        with patch("council.ProviderFactory.get_best_available") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.generate = AsyncMock(return_value=mock_response)
            mock_factory.return_value = mock_provider

            score = await council._check_disagreement({"A": "yes", "B": "no"})
            assert score == 8.0

    @pytest.mark.asyncio
    async def test_disagreement_fallback(self):
        """Test fallback when parsing fails."""
        council = LLMCouncil()

        mock_response = LLMResponse(
            content="invalid json",
            model="test",
            provider="test",
        )

        with patch("council.ProviderFactory.get_best_available") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.generate = AsyncMock(return_value=mock_response)
            mock_factory.return_value = mock_provider

            score = await council._check_disagreement({"A": "yes", "B": "no"})
            assert score == 5.0  # Default fallback
