"""Web search functionality for the Researcher role."""

import httpx
from typing import List, Optional
from dataclasses import dataclass
from duckduckgo_search import DDGS

from config import settings


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str


class WebSearcher:
    """Web search interface with multiple backends."""

    def __init__(self):
        self.serper_key = settings.serper_api_key

    async def search(
        self,
        query: str,
        num_results: int = 5,
    ) -> List[SearchResult]:
        """Search the web using available backend.

        Uses Serper if API key available, falls back to DuckDuckGo.
        """
        if self.serper_key:
            return await self._search_serper(query, num_results)
        return await self._search_duckduckgo(query, num_results)

    async def _search_serper(
        self,
        query: str,
        num_results: int,
    ) -> List[SearchResult]:
        """Search using Serper API."""
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "num": num_results,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

                results = []
                for item in data.get("organic", [])[:num_results]:
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                    ))
                return results

        except Exception as e:
            print(f"Serper search error: {e}, falling back to DuckDuckGo")
            return await self._search_duckduckgo(query, num_results)

    async def _search_duckduckgo(
        self,
        query: str,
        num_results: int,
    ) -> List[SearchResult]:
        """Search using DuckDuckGo (no API key required)."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=num_results))
            )

            return [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                )
                for r in results
            ]

        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return []

    def format_results(self, results: List[SearchResult]) -> str:
        """Format search results for inclusion in prompts."""
        if not results:
            return "No search results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. **{r.title}**\n"
                f"   URL: {r.url}\n"
                f"   {r.snippet}\n"
            )

        return "\n".join(formatted)


# Global searcher instance
web_searcher = WebSearcher()
