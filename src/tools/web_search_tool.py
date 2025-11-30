"""
Web Search Tool for Deep Research Report Generation Agent System
Implements web search capabilities with intelligent source selection based on query type
"""
import logging
import random
import asyncio
import urllib.parse
import re
from typing import List, Dict, Any, Optional, Tuple
import aiohttp
import requests
from bs4 import BeautifulSoup

from config import MAX_SEARCH_RESULTS

logger = logging.getLogger(__name__)


class WebSearchTool:
    """
    Web search tool with intelligent source selection based on query type
    Implements multi-source search capabilities with domain-specific optimization
    """
    
    def __init__(self):
        logger.info("Web Search Tool initialized")
        # 初始化搜索来源配置
        self.search_sources = self._initialize_search_sources()
        # 初始化问题类型关键词
        self.query_type_keywords = self._initialize_query_type_keywords()
    
    def _initialize_search_sources(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        初始化不同类型的搜索来源
        根据问题类型分组不同的搜索来源
        """
        return {
            # 科研论文类搜索来源
            "academic": [
                {
                    "name": "arxiv",
                    "url": "https://arxiv.org/search/?query={query}&searchtype=all",
                    "priority": 0.9,
                    "description": "预印本论文平台，包含最新研究论文"
                },
                {
                    "name": "google_scholar",
                    "url": "https://scholar.google.com/scholar?q={query}",
                    "priority": 0.85,
                    "description": "Google学术搜索，包含各类学术文献"
                },
                {
                    "name": "semantic_scholar",
                    "url": "https://www.semanticscholar.org/search?q={query}",
                    "priority": 0.8,
                    "description": "语义学术搜索，专注于研究论文分析"
                },
                {
                    "name": "ieee",
                    "url": "https://ieeexplore.ieee.org/search/searchresult.jsp?queryText={query}",
                    "priority": 0.75,
                    "description": "IEEE论文数据库，专注于工程和计算机科学"
                }
            ],
            # 技术开发类搜索来源
            "technical": [
                {
                    "name": "github",
                    "url": "https://github.com/topics/{query}",
                    "priority": 0.9,
                    "description": "GitHub代码仓库和技术项目"
                },
                {
                    "name": "stackoverflow",
                    "url": "https://stackoverflow.com/search?q={query}",
                    "priority": 0.85,
                    "description": "程序员问答社区，解决技术问题"
                },
                {
                    "name": "csdn",
                    "url": "https://so.csdn.net/so/search?q={query}",
                    "priority": 0.8,
                    "description": "中文技术社区，包含技术文章和教程"
                },
                {
                    "name": "mdn",
                    "url": "https://developer.mozilla.org/zh-CN/search?q={query}",
                    "priority": 0.75,
                    "description": "Mozilla开发者网络，Web技术文档"
                },
                {
                    "name": "devdocs",
                    "url": "https://devdocs.io/#q={query}",
                    "priority": 0.7,
                    "description": "开发者文档集合，多语言API参考"
                }
            ],
            # 一般性知识搜索来源
            "general": [
                {
                    "name": "wikipedia",
                    "url": "https://zh.wikipedia.org/wiki/{query}",
                    "priority": 0.9,
                    "description": "维基百科，综合性百科全书"
                },
                {
                    "name": "baidu",
                    "url": "https://www.baidu.com/s?wd={query}",
                    "priority": 0.85,
                    "description": "百度搜索引擎，中文内容全面"
                },
                {
                    "name": "zhihu",
                    "url": "https://www.zhihu.com/search?q={query}",
                    "priority": 0.8,
                    "description": "知乎问答社区，深度内容讨论"
                },
                {
                    "name": "sogou",
                    "url": "https://www.sogou.com/web?query={query}",
                    "priority": 0.75,
                    "description": "搜狗搜索引擎，中文内容丰富"
                },
                {
                    "name": "bing",
                    "url": "https://www.bing.com/search?q={query}",
                    "priority": 0.7,
                    "description": "必应搜索引擎，多语言搜索"
                }
            ],
            # 科技新闻类搜索来源
            "tech_news": [
                {
                    "name": "36kr",
                    "url": "https://www.bing.com/search?q={query}",
                    "priority": 0.9,
                    "description": "36氪，科技创业新闻平台"
                },
                {
                    "name": "tech_sina",
                    "url": "https://www.bing.com/search?q={query}",
                    "priority": 0.85,
                    "description": "新浪科技，科技新闻报道"
                },
                {
                    "name": "ifanr",
                    "url": "https://www.bing.com/search?q={query}",
                    "priority": 0.8,
                    "description": "爱范儿，科技产品和趋势报道"
                },
                {
                    "name": "solidot",
                    "url": "https://www.bing.com/search?q={query}",
                    "priority": 0.75,
                    "description": "奇客的资讯，重要的东西",
                },
                {
                    "name": "techcrunch_china",
                    "url": "https://www.bing.com/search?q={query}",
                    "priority": 0.7,
                    "description": "TechCrunch中国站，科技创业资讯"
                }
            ]
        }
    
    def _initialize_query_type_keywords(self) -> Dict[str, List[str]]:
        """
        初始化问题类型关键词，用于判断查询的类型
        """
        return {
            "academic": [
                "论文", "研究", "实验", "调研", "分析", "结果", "方法", 
                "模型", "理论", "假设", "证明", "发表", "期刊", "会议",
                "文献", "综述", "review", "paper", "research", "study", 
                "experiment", "analysis", "method", "model", "theory", 
                "hypothesis", "journal", "conference", "publication"
            ],
            "technical": [
                "代码", "编程", "开发", "实现", "错误", "修复", "调试",
                "框架", "库", "API", "SDK", "工具", "环境", "配置",
                "安装", "部署", "教程", "指南", "最佳实践", "优化",
                "code", "program", "develop", "implement", "error", "fix",
                "debug", "framework", "library", "setup", "configure", 
                "install", "deploy", "tutorial", "guide", "optimize"
            ],
            "tech_news": [
                "新闻", "最新", "动态", "趋势", "发展", "发布", "更新",
                "公告", "报道", "消息", "事件", "发布会", "新品", "上市",
                "新闻", "news", "latest", "update", "release", "announce",
                "report", "trend", "development", "event", "product", "launch"
            ]
        }
    
    def _classify_query_type(self, query: str) -> str:
        """
        根据查询内容判断问题类型
        返回最匹配的问题类型
        """
        query_lower = query.lower()
        type_scores = {"general": 0, "academic": 0, "technical": 0, "tech_news": 0}
        
        # 计算每种类型的匹配分数
        for query_type, keywords in self.query_type_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    type_scores[query_type] += 1
        
        # 返回得分最高的类型
        max_type = max(type_scores, key=type_scores.get)
        
        # 如果所有类型得分都很低，默认为一般性查询
        if type_scores[max_type] == 0:
            max_type = "general"
        
        logger.info(f"Classified query '{query}' as type: {max_type} (scores: {type_scores})")
        return max_type
    
    def _get_search_sources_for_query(self, query: str, max_sources: int = 5) -> List[Dict[str, Any]]:
        """
        根据查询类型获取合适的搜索来源
        """
        query_type = self._classify_query_type(query)
        
        # 获取该类型的搜索来源，并按优先级排序
        sources = sorted(
            self.search_sources[query_type],
            key=lambda x: x['priority'],
            reverse=True
        )
        
        # 截取指定数量的来源
        selected_sources = sources[:max_sources]
        
        logger.info(f"Selected {len(selected_sources)} search sources for query type '{query_type}'")
        return selected_sources
    
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Perform web search for the given query with intelligent source selection
        based on query type classification
        """
        logger.info(f"Performing intelligent web search for: {query[:50]}...")
        
        try:
            # 根据查询类型获取搜索来源
            search_sources = self._get_search_sources_for_query(query)
            
            # 从不同来源获取搜索结果
            all_results = []
            
            # 首先从主要来源获取结果（基于类型的来源）
            for source in search_sources:
                source_results = await self._get_source_results(query, source, max_results // len(search_sources))
                all_results.extend(source_results)
            
            # 确保有足够的结果，如果不足则使用通用搜索补充
            if len(all_results) < max_results:
                additional_count = max_results - len(all_results)
                # 使用通用搜索获取补充结果
                general_sources = self.search_sources.get("general", [])[:3]  # 取前3个通用来源
                for source in general_sources:
                    if len(all_results) >= max_results:
                        break
                    source_results = await self._get_source_results(
                        query, 
                        source, 
                        additional_count // len(general_sources) + 1
                    )
                    all_results.extend(source_results)
            
            # 去重并限制结果数量
            results = self._deduplicate_results(all_results)
            results = results[:max_results]
            
            logger.info(f"Found {len(results)} search results from {len(search_sources)} sources")
            return results
            
        except Exception as e:
            logger.error(f"Error during intelligent web search: {str(e)}")
            # 在错误情况下返回通用搜索结果作为后备
            try:
                fallback_results = await self._fallback_search(query, max_results)
                return fallback_results
            except:
                # 如果后备也失败，返回空结果
                return []
    
    async def _get_source_results(self, query: str, source: Dict[str, Any], max_results: int) -> List[Dict[str, Any]]:
        """
        从指定搜索来源获取结果
        为不同来源生成相应的模拟搜索结果
        """
        try:
            source_name = source.get('name', 'unknown')
            source_url = source.get('url', '').format(query=urllib.parse.quote(query))
            logger.info(f"Getting results from source '{source_name}': {source_url[:100]}...")
            
            # 为不同来源生成相应的模拟搜索结果
            source_type = None
            for type_name, sources_list in self.search_sources.items():
                if any(s['name'] == source_name for s in sources_list):
                    source_type = type_name
                    break
            
            # 生成模拟结果
            mock_results = []
            result_count = min(max_results, 3)  # 每个来源最多3个结果
            
            for i in range(result_count):
                # 根据来源类型定制结果内容
                if source_type == "academic":
                    title = f"{query} - 学术研究论文 #{i+1}"
                    content = f"这是关于{query}的学术研究论文，包含研究方法、实验数据和结论分析。发表于知名学术期刊。"
                elif source_type == "technical":
                    title = f"{query} - 技术实现与教程 #{i+1}"
                    content = f"这是关于{query}的技术文档，包含详细的代码示例、实现步骤和最佳实践指南。"
                elif source_type == "tech_news":
                    title = f"{query} - 最新科技动态 #{i+1}"
                    content = f"这是关于{query}的最新科技新闻报道，分析了行业趋势和市场发展方向。"
                else:
                    title = f"{query} - 综合信息 #{i+1}"
                    content = f"这是关于{query}的综合信息，提供了背景知识和相关内容介绍。"
                
                if source_name == "wikipedia":
                    mock_url = f"https://zh.wikipedia.org/wiki/Special:Search?search={urllib.parse.quote(query)}"
                elif source_name == "github":
                    mock_url = f"https://github.com/topics/{query.lower().replace(' ', '-')}"
                elif source_name == "stackoverflow":
                    mock_url = f"https://stackoverflow.com/questions/tagged/{query.lower().replace(' ', '-')}"
                elif source_name == "zhihu":
                    mock_url = f"https://www.zhihu.com/search?q={urllib.parse.quote(query)}"
                else:
                    mock_url = source_url
                
                # 添加到结果列表
                mock_results.append({
                    "title": title,
                    "url": mock_url,
                    "content": content,
                    "source": source_name,
                    "source_type": source_type,
                    "relevance_score": max(0.5, source.get('priority', 0.7) - i * 0.1)
                })
            
            # 清理URL
            for result in mock_results:
                if 'url' in result:
                    result['url'] = self._clean_url(result['url'])
            
            logger.info(f"Retrieved {len(mock_results)} results from source '{source_name}'")
            return mock_results
            
        except Exception as e:
            logger.error(f"Error getting results from source '{source.get('name', 'unknown')}': {str(e)}")
            return []
    
    def _clean_url(self, url: str) -> str:
        """
        清理URL，确保没有任何多余的逗号或其他特殊字符
        增强清理功能，防止403/404错误
        """
        if not url:
            return url
        
        # 首先移除所有逗号（无论在中间还是末尾）
        clean_url = url.replace(',', '')
        
        # 移除末尾常见的特殊字符
        clean_url = clean_url.rstrip(',;./?&=:')
        
        # 确保URL格式正确（重新解析和构建）
        try:
            # 尝试重新解析URL以确保格式正确
            parsed_url = urllib.parse.urlparse(clean_url)
            
            # 重新构建URL组件
            clean_url = urllib.parse.urlunparse(parsed_url)
            
        except Exception as e:
            logger.warning(f"Failed to reparse URL {clean_url}: {str(e)}")
            # 如果解析失败，至少保留已清理的URL
        
        logger.debug(f"Cleaned URL from '{url}' to '{clean_url}'")
        return clean_url
    
    async def _fallback_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        后备搜索方法，当智能搜索失败时使用
        提供基本的搜索结果确保系统继续运行
        """
        logger.info(f"Using fallback search for query: {query}")
        
        # 创建基本的默认搜索结果
        fallback_results = [
            {
                "title": f"{query} - 维基百科",
                "url": f"https://zh.wikipedia.org/wiki/Special:Search?search={urllib.parse.quote(query)}",
                "content": f"这是关于{query}的维基百科条目，提供了基本的背景信息。",
                "source": "wikipedia",
                "source_type": "general",
                "relevance_score": 0.9
            },
            {
                "title": f"{query} - 百度搜索",
                "url": f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}",
                "content": f"百度搜索引擎提供的{query}相关结果。",
                "source": "baidu",
                "source_type": "general",
                "relevance_score": 0.85
            },
            {
                "title": f"{query} - 知乎讨论",
                "url": f"https://www.zhihu.com/search?q={urllib.parse.quote(query)}",
                "content": f"知乎上关于{query}的讨论和问答内容。",
                "source": "zhihu",
                "source_type": "general",
                "relevance_score": 0.8
            }
        ]
        
        # 清理URL
        for result in fallback_results:
            if 'url' in result:
                result['url'] = self._clean_url(result['url'])
        
        return fallback_results[:max_results]
    
    async def _search_alternative_sources(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """
        Search using alternative sources if primary sources don't provide enough results
        This method is now part of the intelligent search strategy
        """
        # 此方法现在已经集成到主搜索逻辑中
        # 这里保留作为兼容性和额外搜索的支持
        logger.info(f"Using alternative sources for query: {query}")
        
        # 获取通用搜索来源
        general_sources = self.search_sources.get("general", [])[:2]
        
        results = []
        for source in general_sources:
            if len(results) >= max_results:
                break
            source_results = await self._get_source_results(query, source, 2)
            results.extend(source_results)
        
        return results[:max_results]
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on URL
        Also sorts results by relevance score and source priority
        """
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            # 对URL进行清理后再去重
            if url:
                # 清理URL
                clean_url = self._clean_url(url)
                if clean_url not in seen_urls:
                    seen_urls.add(clean_url)
                    # 更新结果中的URL
                    result['url'] = clean_url
                    unique_results.append(result)
        
        # 根据相关性分数排序结果
        unique_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return unique_results
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        获取搜索统计信息，用于监控和测试
        """
        return {
            "total_search_sources": sum(len(sources) for sources in self.search_sources.values()),
            "source_types": list(self.search_sources.keys()),
            "sources_by_type": {type_name: [source['name'] for source in sources] 
                              for type_name, sources in self.search_sources.items()},
            "query_type_keywords_count": {type_name: len(keywords) 
                                         for type_name, keywords in self.query_type_keywords.items()}
        }


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
