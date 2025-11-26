"""
Web Search Tool for Deep Research Report Generation Agent System
Implements web search capabilities based on architecture design
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
import aiohttp
from duckduckgo_search import AsyncDDGS
import requests
from bs4 import BeautifulSoup

from config import MAX_SEARCH_RESULTS

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Web search tool that implements multi-source search capabilities
    Based on the multi-source information integration from architecture
    """
    
    def __init__(self):
        logger.info("Web Search Tool initialized")
    
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform web search for the given query
        Implements the multi-source information integration capability
        """
        logger.info(f"Performing web search for: {query[:50]}...")
        
        try:
            # Use DuckDuckGo search as primary source
            results = await self._search_duckduckgo(query, max_results)
            
            # Enhance with additional search if needed
            if len(results) < max_results:
                additional_results = await self._search_alternative_sources(query, max_results - len(results))
                results.extend(additional_results[:max_results - len(results)])
            
            # Deduplicate results
            results = self._deduplicate_results(results)
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            # Return empty results in case of error
            return []
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        try:
            async with AsyncDDGS() as ddgs:
                results = []
                count = 0
                async for result in ddgs.text(query, max_results=max_results):
                    if count >= max_results:
                        break
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "content": result.get("body", ""),
                        "source": "duckduckgo",
                        "relevance_score": 0.8  # Default score for DuckDuckGo results
                    })
                    count += 1
                return results
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {str(e)}")
            return []
    
    async def _search_alternative_sources(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using alternative sources if primary source fails"""
        # For now, we'll implement a simple alternative search
        # In a real implementation, this might connect to other search APIs
        results = []
        
        # Example: Use requests to search via Google (with caution - may violate ToS)
        # This is just a placeholder implementation
        try:
            # This is a simplified example - in real usage, you'd want to use
            # proper search APIs that allow commercial usage
            pass
        except Exception as e:
            logger.warning(f"Alternative search failed: {str(e)}")
        
        return results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results


class ScraperTool:
    """
    Web scraping tool for extracting content from URLs
    Based on the multi-source information integration from architecture
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        logger.info("Scraper Tool initialized")
    
    async def scrape_url(self, url: str) -> str:
        """
        Scrape content from a given URL
        Implements the information gathering capability from architecture
        """
        logger.info(f"Scraping content from: {url[:50]}...")
        
        try:
            # Use asyncio to run the synchronous requests call
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, self._sync_scrape, url)
            return content
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return f"Error scraping content: {str(e)}"
    
    def _sync_scrape(self, url: str) -> str:
        """Synchronous scraping function to run in executor"""
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text


class ResearchTool:
    """
    Comprehensive research tool combining search and scraping
    """
    
    def __init__(self):
        self.web_search_tool = WebSearchTool()
        self.scraper_tool = ScraperTool()
        logger.info("Research Tool initialized")
    
    async def conduct_research(self, query: str, max_sources: int = 10) -> Dict[str, Any]:
        """
        Conduct comprehensive research using search and scraping
        """
        logger.info(f"Conducting comprehensive research for: {query[:50]}...")
        
        # Perform search
        search_results = await self.web_search_tool.search(query, max_sources)
        
        # Scrape content from top results
        research_data = []
        for result in search_results:
            try:
                content = await self.scraper_tool.scrape_url(result["url"])
                research_data.append({
                    **result,
                    "scraped_content": content,
                    "scraped_at": asyncio.get_event_loop().time()
                })
            except Exception as e:
                logger.warning(f"Failed to scrape {result['url']}: {str(e)}")
                research_data.append({
                    **result,
                    "scraped_content": result.get("content", ""),
                    "scraped_at": asyncio.get_event_loop().time()
                })
        
        return {
            "query": query,
            "search_results": search_results,
            "research_data": research_data,
            "conducted_at": asyncio.get_event_loop().time()
        }