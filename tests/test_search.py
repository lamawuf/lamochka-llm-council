"""Tests for web search functionality."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from search import WebSearcher, SearchResult, web_searcher


class TestSearchResult:
    """Test cases for SearchResult."""

    def test_search_result_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet content",
        )
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet content"


class TestWebSearcher:
    """Test cases for WebSearcher."""

    @pytest.fixture
    def searcher(self):
        """Create a WebSearcher instance."""
        return WebSearcher()

    def test_format_results_empty(self, searcher):
        """Test formatting empty results."""
        formatted = searcher.format_results([])
        assert "No search results found" in formatted

    def test_format_results(self, searcher):
        """Test formatting search results."""
        results = [
            SearchResult(
                title="Result 1",
                url="https://example1.com",
                snippet="First result",
            ),
            SearchResult(
                title="Result 2",
                url="https://example2.com",
                snippet="Second result",
            ),
        ]
        formatted = searcher.format_results(results)

        assert "Result 1" in formatted
        assert "Result 2" in formatted
        assert "https://example1.com" in formatted
        assert "1." in formatted
        assert "2." in formatted

    @pytest.mark.asyncio
    async def test_duckduckgo_search(self, searcher):
        """Test DuckDuckGo search fallback."""
        searcher.serper_key = None  # Force DuckDuckGo

        mock_results = [
            {"title": "DDG Result", "href": "https://ddg.com", "body": "Snippet"},
        ]

        with patch("duckduckgo_search.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = mock_results
            results = await searcher._search_duckduckgo("test query", 5)

            assert len(results) == 1
            assert results[0].title == "DDG Result"

    @pytest.mark.asyncio
    async def test_serper_search_with_key(self, searcher):
        """Test Serper search when API key is available."""
        searcher.serper_key = "test_key"

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic": [
                {"title": "Serper Result", "link": "https://serper.com", "snippet": "Snippet"},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )
            results = await searcher._search_serper("test query", 5)

            assert len(results) == 1
            assert results[0].title == "Serper Result"

    @pytest.mark.asyncio
    async def test_serper_fallback_on_error(self, searcher):
        """Test fallback to DuckDuckGo when Serper fails."""
        searcher.serper_key = "test_key"

        ddg_results = [
            {"title": "DDG Fallback", "href": "https://ddg.com", "body": "Fallback"},
        ]

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post.side_effect = Exception("Error")

            with patch("duckduckgo_search.DDGS") as mock_ddgs:
                mock_ddgs.return_value.text.return_value = ddg_results
                results = await searcher._search_serper("test query", 5)

                # Should fall back to DuckDuckGo
                assert len(results) == 1
                assert results[0].title == "DDG Fallback"


class TestGlobalSearcher:
    """Test global searcher instance."""

    def test_global_instance_exists(self):
        """Test global web_searcher instance exists."""
        assert web_searcher is not None
        assert isinstance(web_searcher, WebSearcher)
