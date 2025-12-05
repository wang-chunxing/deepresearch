"""
Web Scraper Tool for Deep Research Report Generation Agent System
Implements web page content extraction capabilities
"""
import asyncio
import logging
from typing import Dict, Any, Optional
import aiohttp
from bs4 import BeautifulSoup
import time
import urllib.parse

# 尝试导入brotli库，用于处理br压缩的响应
_br_available = False
try:
    import brotli
    _br_available = True
except ImportError:
    logging.info("Brotli library not available, will skip br decompression")

logger = logging.getLogger(__name__)


class ScraperTool:
    """
    Web scraper tool that implements content extraction from web pages
    Based on the multi-source information integration from architecture
    """
    
    def __init__(self):
        logger.info("Web Scraper Tool initialized")
        self.timeout = aiohttp.ClientTimeout(total=30)
        self._do_not_scrape_domains = {"www.sciencedirect.com", "sciencedirect.com", "www.mdpi.com", "mdpi.com"}
        self._min_delay = 0.6
    
    def _get_random_user_agent(self) -> str:
        """返回随机的User-Agent以避免被识别为爬虫"""
        user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        import random
        return random.choice(user_agents)
    
    async def scrape(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from the given URL
        Implements the content extraction capability
        """
        logger.info(f"Scraping content from URL: {url}")
        
        try:
            # 使用aiohttp异步获取网页内容
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # 添加更完善的请求头，模拟真实浏览器行为
                headers = {
                    'User-Agent': self._get_random_user_agent(),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                    'Cache-Control': 'max-age=0'
                }
                try:
                    parsed = urllib.parse.urlparse(url)
                    origin = f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme and parsed.netloc else None
                    if origin:
                        headers['Referer'] = origin
                except Exception:
                    pass
                
                # 轻微延迟，减少触发速率限制
                await asyncio.sleep(self._min_delay)
                async with session.get(url, headers=headers) as response:
                    # 根据不同状态码提供更详细的错误信息
                    if response.status == 403:
                        logger.warning(f"Access forbidden (403) when fetching URL: {url}. Possible reasons: IP blocked, missing permissions, or User-Agent rejected")
                        try:
                            fallback = f"https://www.bing.com/search?q={urllib.parse.quote(url)}"
                            async with session.get(fallback, headers=headers) as fb:
                                html_content = await fb.text()
                                soup = BeautifulSoup(html_content, 'html.parser')
                                for script in soup(['script', 'style']):
                                    script.decompose()
                                text = '\n'.join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
                                return {
                                    'url': url,
                                    'success': True,
                                    'title': 'Fallback Search Result',
                                    'content': text,
                                    'timestamp': time.time()
                                }
                        except Exception:
                            return {
                                'url': url,
                                'success': False,
                                'error': f"Access forbidden (403): The server denied access to this resource. This might be due to bot protection.",
                                'content': None,
                                'status_code': 403
                            }
                    elif response.status == 404:
                        logger.warning(f"URL not found (404): {url}")
                        try:
                            fallback = f"https://www.bing.com/search?q={urllib.parse.quote(url)}"
                            async with session.get(fallback, headers=headers) as fb:
                                html_content = await fb.text()
                                soup = BeautifulSoup(html_content, 'html.parser')
                                for script in soup(['script', 'style']):
                                    script.decompose()
                                text = '\n'.join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
                                return {
                                    'url': url,
                                    'success': True,
                                    'title': 'Fallback Search Result',
                                    'content': text,
                                    'timestamp': time.time()
                                }
                        except Exception:
                            return {
                                'url': url,
                                'success': False,
                                'error': f"Page not found (404): The requested URL does not exist.",
                                'content': None,
                                'status_code': 404
                            }
                    elif response.status in (429, 503):
                        # 处理速率限制和服务不可用
                        retry_after = response.headers.get('Retry-After', 'unknown')
                        logger.error(f"Rate limited or service unavailable ({response.status}) for URL: {url}. Retry-After: {retry_after}")
                        return {
                            'url': url,
                            'success': False,
                            'error': f"Too many requests or service unavailable ({response.status}). Please try again later. Retry-After: {retry_after}",
                            'content': None,
                            'status_code': response.status,
                            'retry_after': retry_after
                        }
                    elif response.status >= 500:
                        # 服务器错误
                        logger.error(f"Server error ({response.status}) when fetching URL: {url}")
                        return {
                            'url': url,
                            'success': False,
                            'error': f"Server error ({response.status}): The server encountered an internal error.",
                            'content': None,
                            'status_code': response.status
                        }
                    elif response.status != 200:
                        # 其他非成功状态码
                        logger.error(f"Failed to fetch URL: {url}, status code: {response.status}")
                        return {
                            'url': url,
                            'success': False,
                            'error': f"HTTP status code: {response.status}",
                            'content': None,
                            'status_code': response.status
                        }
                    
                    # 获取网页内容
                    content_encoding = response.headers.get('Content-Encoding', '').lower()
                    
                    if 'br' in content_encoding and _br_available:
                        # 如果是br压缩且brotli库可用，手动解压
                        try:
                            raw_content = await response.read()
                            html_content = brotli.decompress(raw_content).decode('utf-8')
                            logger.debug(f"Successfully decompressed Brotli content from {url}")
                        except Exception as e:
                            logger.error(f"Failed to decompress Brotli content from {url}: {str(e)}")
                            # 如果解压失败，尝试使用默认的text()方法
                            html_content = await response.text()
                    else:
                        # 使用默认方法获取内容
                        html_content = await response.text()
                    
                    # 使用BeautifulSoup解析HTML（200 成功时继续）
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # 提取标题
                    title = soup.title.string if soup.title else 'No title'
                    
                    # 提取正文内容（移除脚本和样式标签）
                    for script in soup(['script', 'style']):
                        script.decompose()
                    
                    # 获取文本内容
                    text = soup.get_text()
                    
                    # 清理文本（移除多余的空白字符）
                    clean_text = '\n'.join([line.strip() for line in text.splitlines() if line.strip()])
                    
                    logger.info(f"Successfully scraped content from {url}")
                    
                    return {
                        'url': url,
                        'success': True,
                        'title': title,
                        'content': clean_text,
                        'timestamp': time.time()
                    }
                    
        except aiohttp.ClientError as e:
            # 捕获所有aiohttp客户端错误
            error_type = type(e).__name__
            if 'Timeout' in error_type or 'Connection' in error_type:
                logger.warning(f"AIOHTTP error ({error_type}) while scraping {url}: {str(e)}")
            else:
                logger.error(f"AIOHTTP error ({error_type}) while scraping {url}: {str(e)}")
            
            # 特别处理常见的客户端错误
            if 'SSL' in error_type:
                error_msg = f"SSL error: There was a problem with the SSL certificate. {str(e)}"
            elif 'Timeout' in error_type:
                error_msg = f"Request timeout: The server took too long to respond. {str(e)}"
            elif 'Connection' in error_type:
                error_msg = f"Connection error: Could not establish connection to the server. {str(e)}"
            else:
                error_msg = f"Client error: {str(e)}"
            
            return {
                'url': url,
                'success': False,
                'error': error_msg,
                'error_type': error_type,
                'content': None,
                'status_code': None  # 添加status_code键
            }
        except Exception as e:
            # 捕获所有其他未预期的错误
            error_type = type(e).__name__
            logger.error(f"Unexpected error ({error_type}) while scraping {url}: {str(e)}")
            return {
                'url': url,
                'success': False,
                'error': f"Unexpected error: {str(e)}",
                'error_type': error_type,
                'content': None,
                'status_code': None  # 添加status_code键
            }
    
    async def scrape_url(self, url: str) -> str:
        """
        兼容方法：调用scrape并返回content字段
        用于兼容research_agent.py中的调用
        """
        # 清理URL（确保没有逗号等问题）
        import re
        clean_url = url.replace(',', '')
        clean_url = re.sub(r'[,;./\\:]+$', '', clean_url)
        
        result = await self.scrape(clean_url)
        if result.get('success'):
            return result.get('content', '')
        else:
            logger.error(f"Failed to scrape URL with scrape_url: {clean_url}, error: {result.get('error')}")
            return f"Error scraping content: {result.get('error')}" 

    def should_scrape(self, url: str) -> bool:
        try:
            parsed = urllib.parse.urlparse(url)
            host = parsed.netloc.lower()
            path = parsed.path.lower()
            if host in self._do_not_scrape_domains:
                return False
            if '/search' in path:
                return False
            return True
        except Exception:
            return True
    
    async def scrape_multiple(self, urls: list, delay: float = 1.0, retry_on_error: bool = True, max_retries: int = 2) -> list:
        """
        Scrape multiple URLs with a delay between requests to avoid rate limiting
        
        Args:
            urls: List of URLs to scrape
            delay: Delay between requests in seconds
            retry_on_error: Whether to retry failed requests
            max_retries: Maximum number of retries for failed requests
            
        Returns:
            List of scraping results
        """
        results = []
        
        for url in urls:
            retries = 0
            success = False
            
            while not success and retries <= max_retries:
                result = await self.scrape(url)
                
                # 检查是否需要重试
                should_retry = retry_on_error and not result.get('success') and retries < max_retries
                
                # 决定是否重试（只对特定错误类型进行重试）
                status_code = result.get('status_code')
                error_type = result.get('error_type')
                
                # 只对特定错误进行重试：服务器错误(5xx)、超时、连接错误、429(太多请求)
                if should_retry and (
                    (status_code and status_code >= 500) or
                    (status_code == 429) or
                    (error_type and ('Timeout' in error_type or 'Connection' in error_type))
                ):
                    retries += 1
                    wait_time = delay * (2 ** (retries - 1))  # 指数退避策略
                    logger.info(f"Retrying {url} ({retries}/{max_retries}) after {wait_time:.2f}s due to error: {result.get('error')}")
                    await asyncio.sleep(wait_time)
                else:
                    # 不再重试，添加结果
                    results.append(result)
                    success = True
                    
            # 添加延迟，避免对服务器造成过大压力
            if url != urls[-1]:  # 不为最后一个URL时添加延迟
                await asyncio.sleep(delay)
        
        # 统计结果
        success_count = sum(1 for r in results if r.get('success'))
        logger.info(f"Scraped {len(urls)} URLs: {success_count} successful, {len(results) - success_count} failed")
        
        return results
