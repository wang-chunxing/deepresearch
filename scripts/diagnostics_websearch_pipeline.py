import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tools.web_search_tool import WebSearchTool
from src.tools.scraper_tool import ScraperTool
from src.generation.report_generator import ReportGenerator
mm_available = True
try:
    from src.memory.memory_manager import MemoryManager
except Exception:
    mm_available = False
import numpy as np

def ts(label: str) -> str:
    return f"[{datetime.now().isoformat()}] {label}"

def split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    buf = []
    l = 0
    for p in parts:
        if l + len(p) <= 1200:
            buf.append(p)
            l += len(p)
        else:
            if buf:
                chunks.append('\n'.join(buf))
            buf = [p]
            l = len(p)
    if buf:
        chunks.append('\n'.join(buf))
    return chunks

def embed_chunks(chunks: List[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        vecs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
        return vecs
    except Exception:
        arr = []
        for c in chunks:
            h = hash(c)
            v = np.array([(h >> i) & 255 for i in range(0, 256, 8)], dtype=np.float32)
            v = v / (np.linalg.norm(v) + 1e-9)
            arr.append(v)
        return np.stack(arr)

async def run_query(query: str) -> Dict[str, Any]:
    web = WebSearchTool()
    scraper = ScraperTool()
    mm = MemoryManager() if mm_available else None
    out: Dict[str, Any] = {"query": query, "stages": []}
    t0 = time.time()
    out["stages"].append({"t": datetime.now().isoformat(), "stage": "start", "query": query})
    srcs = web._get_search_sources_for_query(query)
    raw = []
    for s in srcs:
        rs = await web._get_source_results(query, s, 3)
        raw.append({"source": s.get("name"), "url": s.get("url"), "results": rs})
    out["stages"].append({"t": datetime.now().isoformat(), "stage": "raw_source_results", "sources": [x.get("source") for x in raw], "raw": raw})
    results = await web.search(query, max_results=10)
    out["stages"].append({"t": datetime.now().isoformat(), "stage": "search_results", "count": len(results), "results": results})
    scraped: List[Dict[str, Any]] = []
    for r in results:
        c = await scraper.scrape_url(r["url"])
        scraped.append({"source": r["source"], "url": r["url"], "content": c})
    out["stages"].append({"t": datetime.now().isoformat(), "stage": "scraped", "count": len(scraped)})
    splits: List[Dict[str, Any]] = []
    for s in scraped:
        chs = split_paragraphs(s["content"])
        splits.append({"url": s["url"], "chunks": chs})
    out["stages"].append({"t": datetime.now().isoformat(), "stage": "split", "doc_count": len(splits), "total_chunks": sum(len(x["chunks"]) for x in splits)})
    all_chunks = []
    for sp in splits:
        for c in sp["chunks"]:
            all_chunks.append(c)
    vecs = embed_chunks(all_chunks) if all_chunks else np.zeros((0,32))
    out["stages"].append({"t": datetime.now().isoformat(), "stage": "vectorize", "vec_shape": list(vecs.shape)})
    if mm is not None:
        for s in scraped:
            await mm.add_to_external_memory(s["content"], {"type": "diagnostics_source", "source_url": s["url"]})
        stats = await mm.get_memory_stats()
        out["stages"].append({"t": datetime.now().isoformat(), "stage": "vector_store_stats", "stats": stats})
    else:
        out["stages"].append({"t": datetime.now().isoformat(), "stage": "vector_store_stats", "stats": {"external_memory": {"entries": 0}, "note": "memory_manager not available"}})
    sources = []
    information = []
    for i, r in enumerate(results):
        src = {"title": r.get("title", ""), "url": r.get("url", ""), "summary": r.get("content", ""), "relevance_score": r.get("relevance_score", 0.0)}
        sources.append(src)
    for s in scraped:
        information.append({"source": {"url": s["url"], "title": s.get("source", "")}, "content": s["content"], "extracted_at": datetime.now().isoformat()})
    report_data = {
        "query": query,
        "research_plan": {"query": query, "depth": "diagnostics", "steps": ["search", "scrape", "split", "vectorize"], "planned_at": datetime.now().isoformat()},
        "gathered_information": {"sources": sources, "information": information},
        "analysis": {"summary": "diagnostics run", "key_findings": [], "themes": [], "contradictions": [], "gaps": []},
        "validated_findings": {"confidence_level": 0.5},
        "sources": sources,
        "timestamp": datetime.now().isoformat()
    }
    rg = ReportGenerator()
    rendered = await rg.generate_report(report_data, format_type="markdown", include_sources=True)
    out["stages"].append({"t": datetime.now().isoformat(), "stage": "template_render", "format": "markdown", "length": len(rendered)})
    out["stages"].append({"t": datetime.now().isoformat(), "stage": "model_call_plan", "phases": [
        {"phase": "intent_recognition", "agent": "planning_agent", "provider": "configured", "input_size": sum(len(x.get("summary","")) for x in sources)},
        {"phase": "query_optimization", "agent": "research_agent", "strategy": "keyword_cleanup"},
        {"phase": "segmented_processing", "agent": "analysis_agent", "chunks": sum(len(sp["chunks"]) for sp in [{"chunks": split_paragraphs(i["content"]) } for i in information])}
    ], "note": "diagnostics log only"})
    out["stages"].append({"t": datetime.now().isoformat(), "stage": "end", "elapsed": round(time.time() - t0, 3)})
    return out

async def main():
    queries = [
        "测试任务",
        "langchain 1.0 最新关键技术解读"
    ]
    print(ts("diagnostics.start"))
    for q in queries:
        res = await run_query(q)
        print(ts("diagnostics.query.begin"), q)
        for st in res["stages"]:
            print(ts(f"stage.{st['stage']}"), st)
        print(ts("diagnostics.query.end"), q)
    print(ts("diagnostics.finish"))

if __name__ == "__main__":
    asyncio.run(main())
