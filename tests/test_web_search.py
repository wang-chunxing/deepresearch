"""
测试Web搜索功能，包括智能搜索来源选择和查询类型分类
"""
import unittest
import asyncio
from typing import List, Dict, Any
import logging

# 设置测试日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入要测试的模块
try:
    from src.tools.web_search_tool import WebSearchTool
except ImportError:
    logger.error("无法导入WebSearchTool模块，请检查路径")
    raise


class TestWebSearchTool(unittest.TestCase):
    """
    测试WebSearchTool的功能，包括智能搜索来源选择和查询类型分类
    """
    
    def setUp(self):
        """
        在每个测试用例前设置测试环境
        """
        logger.info("设置WebSearchTool测试环境")
        self.search_tool = WebSearchTool()
    
    def test_query_type_classification(self):
        """
        测试问题类型分类功能
        验证系统能够正确识别不同类型的查询
        """
        logger.info("测试查询类型分类功能")
        
        # 测试学术类查询
        academic_query = "深度学习模型在自然语言处理中的应用研究论文"
        academic_type = self.search_tool._classify_query_type(academic_query)
        self.assertEqual(academic_type, "academic", 
                         f"学术查询 '{academic_query}' 应该被分类为 'academic'，实际为 '{academic_type}'")
        
        # 测试技术类查询
        technical_query = "Python代码实现异步爬虫框架的最佳实践"
        technical_type = self.search_tool._classify_query_type(technical_query)
        self.assertEqual(technical_type, "technical", 
                         f"技术查询 '{technical_query}' 应该被分类为 'technical'，实际为 '{technical_type}'")
        
        # 测试科技新闻类查询
        news_query = "最新AI大模型发布公告和技术趋势"
        news_type = self.search_tool._classify_query_type(news_query)
        self.assertEqual(news_type, "tech_news", 
                         f"新闻查询 '{news_query}' 应该被分类为 'tech_news'，实际为 '{news_type}'")
        
        # 测试通用类查询
        general_query = "北京旅游攻略和景点介绍"
        general_type = self.search_tool._classify_query_type(general_query)
        self.assertEqual(general_type, "general", 
                         f"通用查询 '{general_query}' 应该被分类为 'general'，实际为 '{general_type}'")
        
        logger.info("查询类型分类测试通过")
    
    def test_search_sources_selection(self):
        """
        测试搜索来源选择功能
        验证系统能够根据查询类型选择合适的搜索来源
        """
        logger.info("测试搜索来源选择功能")
        
        # 测试学术类查询的来源选择
        academic_query = "量子计算研究进展论文"
        academic_sources = self.search_tool._get_search_sources_for_query(academic_query, max_sources=3)
        
        self.assertTrue(len(academic_sources) > 0, "学术查询应该返回搜索来源")
        # 检查返回的来源是否包含学术相关来源
        academic_source_names = [source['name'] for source in academic_sources]
        self.assertTrue(any(name in ['arxiv', 'google_scholar', 'semantic_scholar'] 
                          for name in academic_source_names), 
                       f"学术查询返回的来源 {academic_source_names} 应包含学术相关来源")
        
        # 测试技术类查询的来源选择
        technical_query = "React框架性能优化教程"
        technical_sources = self.search_tool._get_search_sources_for_query(technical_query, max_sources=3)
        
        self.assertTrue(len(technical_sources) > 0, "技术查询应该返回搜索来源")
        # 检查返回的来源是否包含技术相关来源
        technical_source_names = [source['name'] for source in technical_sources]
        self.assertTrue(any(name in ['github', 'stackoverflow', 'csdn'] 
                          for name in technical_source_names), 
                       f"技术查询返回的来源 {technical_source_names} 应包含技术相关来源")
        
        logger.info("搜索来源选择测试通过")
    
    def test_url_cleaning(self):
        """
        测试URL清理功能
        验证系统能够正确清理URL中的特殊字符
        """
        logger.info("测试URL清理功能")
        
        # 测试用例1：末尾包含特殊字符的URL
        dirty_url1 = "https://example.com/search,;./"
        clean_url1 = self.search_tool._clean_url(dirty_url1)
        self.assertEqual(clean_url1, "https://example.com/search", 
                         f"URL清理失败，应该去除末尾特殊字符，结果为: {clean_url1}")
        
        # 测试用例2：中间包含逗号的URL
        dirty_url2 = "https://example.com/search?q=term1,term2"
        clean_url2 = self.search_tool._clean_url(dirty_url2)
        self.assertEqual(clean_url2, "https://example.com/search?q=term1term2", 
                         f"URL清理失败，应该去除中间的逗号，结果为: {clean_url2}")
        
        # 测试用例3：正常URL不应被修改
        normal_url = "https://example.com/search?q=normal"
        clean_normal_url = self.search_tool._clean_url(normal_url)
        self.assertEqual(clean_normal_url, normal_url, 
                         f"正常URL不应被修改，结果为: {clean_normal_url}")
        
        # 测试用例4：空URL处理
        empty_url = ""
        clean_empty_url = self.search_tool._clean_url(empty_url)
        self.assertEqual(clean_empty_url, "", "空URL应该返回空字符串")
        
        # 测试用例5：增强的URL清理 - 末尾包含多种特殊字符
        enhanced_dirty_url1 = "https://example.com/path;.,/?&="
        enhanced_clean_url1 = self.search_tool._clean_url(enhanced_dirty_url1)
        self.assertEqual(enhanced_clean_url1, "https://example.com/path",
                        f"增强URL清理失败，应该去除更多种类的末尾特殊字符，结果为: {enhanced_clean_url1}")
        
        # 测试用例6：增强的URL清理 - 查询参数中的末尾逗号
        enhanced_dirty_url2 = "https://example.com/path?param=value,"
        enhanced_clean_url2 = self.search_tool._clean_url(enhanced_dirty_url2)
        self.assertEqual(enhanced_clean_url2, "https://example.com/path?param=value",
                        f"增强URL清理失败，应该去除查询参数中的末尾逗号，结果为: {enhanced_clean_url2}")
        
        # 测试用例7：增强的URL清理 - 路径中的末尾逗号
        enhanced_dirty_url3 = "https://example.com/path,/subpath"
        enhanced_clean_url3 = self.search_tool._clean_url(enhanced_dirty_url3)
        self.assertEqual(enhanced_clean_url3, "https://example.com/path/subpath",
                        f"增强URL清理失败，应该去除路径中的末尾逗号，结果为: {enhanced_clean_url3}")
        
        # 测试用例8：实际错误日志中的URL修复 - 新浪科技
        sina_error_url = "https://tech.sina.com.cn/search?q=langchain%201.0%20%E6%9C%80%E6%96%B0%E5%85%B3%E9%94%AE%E6%8A%80%E6%9C%AF%E8%A7%A3%E8%AF%BB?result=1,"
        sina_clean_url = self.search_tool._clean_url(sina_error_url)
        self.assertEqual(sina_clean_url, "https://tech.sina.com.cn/search?q=langchain%201.0%20%E6%9C%80%E6%96%B0%E5%85%B3%E9%94%AE%E6%8A%80%E6%9C%AF%E8%A7%A3%E8%AF%BB?result=1",
                        f"新浪科技错误URL清理失败，结果为: {sina_clean_url}")
        
        # 测试用例9：实际错误日志中的URL修复 - 爱范儿
        ifanr_error_url = "https://www.ifanr.com/search/langchain%201.0%20%E6%9C%80%E6%96%B0%E5%85%B3%E9%94%AE%E6%8A%80%E6%9C%AF%E8%A7%A3%E8%AF%BB?result=1,"
        ifanr_clean_url = self.search_tool._clean_url(ifanr_error_url)
        self.assertEqual(ifanr_clean_url, "https://www.ifanr.com/search/langchain%201.0%20%E6%9C%80%E6%96%B0%E5%85%B3%E9%94%AE%E6%8A%80%E6%9C%AF%E8%A7%A3%E8%AF%BB?result=1",
                        f"爱范儿错误URL清理失败，结果为: {ifanr_clean_url}")
        
        # 测试用例10：实际错误日志中的URL修复 - Solidot
        solidot_error_url = "https://www.solidot.org/search.pl?q=langchain%201.0%20%E6%9C%80%E6%96%B0%E5%85%B3%E9%94%AE%E6%8A%80%E6%9C%AF%E8%A7%A3%E8%AF%BB?result=2,"
        solidot_clean_url = self.search_tool._clean_url(solidot_error_url)
        self.assertEqual(solidot_clean_url, "https://www.solidot.org/search.pl?q=langchain%201.0%20%E6%9C%80%E6%96%B0%E5%85%B3%E9%94%AE%E6%8A%80%E6%9C%AF%E8%A7%A3%E8%AF%BB?result=2",
                        f"Solidot错误URL清理失败，结果为: {solidot_clean_url}")
        
        logger.info("URL清理功能测试通过")
    
    def test_search_statistics(self):
        """
        测试搜索统计功能
        验证系统能够提供正确的统计信息
        """
        logger.info("测试搜索统计功能")
        
        stats = self.search_tool.get_search_statistics()
        
        # 验证统计信息包含所有必要的键
        self.assertIn("total_search_sources", stats, "统计信息应包含总搜索来源数")
        self.assertIn("source_types", stats, "统计信息应包含来源类型")
        self.assertIn("sources_by_type", stats, "统计信息应包含按类型分组的来源")
        self.assertIn("query_type_keywords_count", stats, "统计信息应包含关键词数量")
        
        # 验证来源类型包含所有预期类型
        expected_types = ["academic", "technical", "general", "tech_news"]
        for expected_type in expected_types:
            self.assertIn(expected_type, stats["source_types"], 
                         f"来源类型应包含 '{expected_type}'")
        
        logger.info("搜索统计功能测试通过")


class TestWebSearchToolAsync(unittest.TestCase):
    """
    测试WebSearchTool的异步功能
    """
    
    def setUp(self):
        """
        在每个测试用例前设置测试环境
        """
        logger.info("设置异步WebSearchTool测试环境")
        self.search_tool = WebSearchTool()
    
    def run_async_test(self, coro):
        """
        运行异步测试协程
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    
    def test_academic_search(self):
        """
        测试学术类搜索功能
        验证系统能够针对学术查询返回相关结果
        """
        logger.info("测试学术类搜索功能")
        
        query = "机器学习算法研究论文"
        results = self.run_async_test(self.search_tool.search(query, max_results=5))
        
        self.assertTrue(len(results) > 0, "学术搜索应返回结果")
        
        # 检查结果中是否包含学术相关的来源类型
        academic_sources = [r for r in results if r.get('source_type') == 'academic']
        self.assertTrue(len(academic_sources) > 0, 
                       f"学术搜索应返回学术相关结果，结果为: {[r.get('source_type') for r in results]}")
        
        logger.info("学术类搜索功能测试通过")
    
    def test_technical_search(self):
        """
        测试技术类搜索功能
        验证系统能够针对技术查询返回相关结果
        """
        logger.info("测试技术类搜索功能")
        
        query = "Python Django框架开发教程"
        results = self.run_async_test(self.search_tool.search(query, max_results=5))
        
        self.assertTrue(len(results) > 0, "技术搜索应返回结果")
        
        # 检查结果中是否包含技术相关的来源类型
        technical_sources = [r for r in results if r.get('source_type') == 'technical']
        self.assertTrue(len(technical_sources) > 0, 
                       f"技术搜索应返回技术相关结果，结果为: {[r.get('source_type') for r in results]}")
        
        logger.info("技术类搜索功能测试通过")
    
    def test_news_search(self):
        """
        测试科技新闻类搜索功能
        验证系统能够针对科技新闻查询返回相关结果
        """
        logger.info("测试科技新闻类搜索功能")
        
        query = "最新AI技术发布新闻"
        results = self.run_async_test(self.search_tool.search(query, max_results=5))
        
        self.assertTrue(len(results) > 0, "新闻搜索应返回结果")
        
        # 检查结果中是否包含新闻相关的来源类型
        news_sources = [r for r in results if r.get('source_type') == 'tech_news']
        self.assertTrue(len(news_sources) > 0, 
                       f"新闻搜索应返回新闻相关结果，结果为: {[r.get('source_type') for r in results]}")
        
        logger.info("科技新闻类搜索功能测试通过")
    
    def test_general_search(self):
        """
        测试通用类搜索功能
        验证系统能够针对通用查询返回相关结果
        """
        logger.info("测试通用类搜索功能")
        
        query = "旅游景点推荐"
        results = self.run_async_test(self.search_tool.search(query, max_results=5))
        
        self.assertTrue(len(results) > 0, "通用搜索应返回结果")
        
        # 检查结果的URL是否都经过清理
        for result in results:
            self.assertNotIn(',', result.get('url', ''), 
                            f"结果URL应经过清理，不应包含逗号: {result.get('url')}")
        
        logger.info("通用类搜索功能测试通过")
    
    def test_search_result_format(self):
        """
        测试搜索结果格式
        验证所有搜索结果都包含必要的字段
        """
        logger.info("测试搜索结果格式")
        
        query = "测试搜索结果格式"
        results = self.run_async_test(self.search_tool.search(query, max_results=3))
        
        # 检查每个结果是否包含必要的字段
        required_fields = ["title", "url", "content", "source", "relevance_score"]
        
        for i, result in enumerate(results):
            for field in required_fields:
                self.assertIn(field, result, 
                             f"搜索结果 #{i+1} 应包含字段 '{field}'，实际字段: {list(result.keys())}")
        
        logger.info("搜索结果格式测试通过")
    
    def test_result_deduplication(self):
        """
        测试结果去重功能
        验证系统能够正确去除重复结果
        """
        logger.info("测试结果去重功能")
        
        # 创建包含重复URL的测试结果
        test_results = [
            {"title": "结果1", "url": "https://example.com/1", "content": "内容1", "source": "source1", "relevance_score": 0.9},
            {"title": "结果2", "url": "https://example.com/1,./", "content": "内容2", "source": "source2", "relevance_score": 0.8},
            {"title": "结果3", "url": "https://example.com/2", "content": "内容3", "source": "source3", "relevance_score": 0.7}
        ]
        
        # 去重后应该只剩下2个结果
        deduplicated_results = self.search_tool._deduplicate_results(test_results)
        self.assertEqual(len(deduplicated_results), 2, 
                         f"去重后应有2个结果，实际有{len(deduplicated_results)}个")
        
        # 检查去重后的URL是否已经清理
        urls = [r["url"] for r in deduplicated_results]
        self.assertEqual(urls, ["https://example.com/1", "https://example.com/2"], 
                         f"去重后的URL应已清理，实际为: {urls}")
        
        logger.info("结果去重功能测试通过")
    
    def test_fallback_search(self):
        """
        测试后备搜索功能
        验证系统在主要搜索失败时能够返回后备结果
        """
        logger.info("测试后备搜索功能")
        
        query = "测试后备搜索"
        fallback_results = self.run_async_test(self.search_tool._fallback_search(query, max_results=3))
        
        self.assertTrue(len(fallback_results) > 0, "后备搜索应返回结果")
        
        # 检查后备结果是否包含必要字段
        for result in fallback_results:
            self.assertIn("title", result, "后备搜索结果应包含标题")
            self.assertIn("url", result, "后备搜索结果应包含URL")
        
        logger.info("后备搜索功能测试通过")


if __name__ == "__main__":
    unittest.main()
