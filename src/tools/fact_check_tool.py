import asyncio
import logging
from typing import List, Dict, Any

from .web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


class FactCheckTool:
    def __init__(self):
        self.search_tool = WebSearchTool()

    async def check_claims(self, claims: List[str], context_sources: List[Dict[str, Any]], max_checks: int = 3) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        supported = 0
        checked = 0
        claims = [c for c in claims if c][:max_checks]
        async def verify(claim: str) -> Dict[str, Any]:
            hits = await self.search_tool.search(claim, max_results=5)
            citations = []
            for h in hits[:5]:
                citations.append({"title": h.get("title", ""), "url": h.get("url", ""), "source": h.get("source", ""), "relevance": h.get("relevance_score", 0.0)})
            ok = len(citations) >= 2
            return {"claim": claim, "supported": ok, "citations": citations}
        for claim in claims:
            r = await verify(claim)
            results.append(r)
            checked += 1
            if r.get("supported"):
                supported += 1
        score = supported / max(1, checked)
        status = "passed" if score >= 0.6 else "needs_review"
        return {"status": status, "score": round(score, 2), "details": results}

