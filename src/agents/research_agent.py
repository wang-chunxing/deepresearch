"""
Research Agent for Deep Research Report Generation Agent System
Implements the reasoning layer based on multi-agent collaboration and dynamic planning
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from config import LLM_BASE_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from src.memory.memory_manager import MemoryManager
from src.generation.report_generator import ReportGenerator
from src.tools.web_search_tool import WebSearchTool
from src.tools.scraper_tool import ScraperTool

logger = logging.getLogger(__name__)


class ResearchAgent:
    """
    主要研究代理，协调整个研究过程
    基于多代理协作实现推理层
    """
    
    def __init__(self, memory_manager: MemoryManager, report_generator: ReportGenerator):
        # 初始化内存管理器和报告生成器
        self.memory_manager = memory_manager
        self.report_generator = report_generator
        
        # 初始化大语言模型
        self.llm = ChatOpenAI(
            model_name=LLM_BASE_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        # 初始化工具
        self.web_search_tool = WebSearchTool()
        self.scraper_tool = ScraperTool()
        
        # 初始化专门代理（通过不同提示/角色模拟）
        self.planning_agent = self._create_planning_agent()
        self.research_agent = self._create_research_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.validation_agent = self._create_validation_agent()
        
        logger.info("研究代理已使用专门代理组件初始化")
    
    def _create_planning_agent(self):
        """创建规划代理组件"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a research planning specialist. Your role is to break down complex research queries into manageable steps and create a research plan."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | self.llm
    
    def _create_research_agent(self):
        """创建研究代理组件"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a research specialist. Your role is to gather relevant information from various sources based on the research plan."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | self.llm
    
    def _create_analysis_agent(self):
        """创建分析代理组件"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an analysis specialist. Your role is to synthesize gathered information, identify patterns, and draw insights."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | self.llm
    
    def _create_validation_agent(self):
        """创建验证代理组件"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a validation specialist. Your role is to verify the accuracy and reliability of information and conclusions."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | self.llm
    
    async def research(self, query: str, max_sources: int = 10, depth: str = "comprehensive") -> Dict[str, Any]:
        """
        对给定查询进行研究
        实现多代理协作和推理链构建
        """
        logger.info(f"开始研究查询: {query[:50]}...")
        
        # 将查询添加到对话历史中
        await self.memory_manager.add_to_conversation_history("user", query)
        
        # 步骤1：规划研究
        research_plan = await self._plan_research(query, depth)
        logger.debug("研究计划已创建")
        
        # 步骤2：执行研究计划
        gathered_info = await self._execute_research_plan(research_plan, max_sources)
        logger.debug(f"收集到 {len(gathered_info.get('sources', []))} 个来源")
        
        # 步骤3：分析信息
        analysis_result = await self._analyze_information(gathered_info, query)
        logger.debug("信息分析完成")
        
        # 步骤4：验证结果
        validated_result = await self._validate_findings(analysis_result, gathered_info)
        logger.debug("结果验证完成")
        
        # 合并结果
        result = {
            "query": query,
            "research_plan": research_plan,
            "gathered_information": gathered_info,
            "analysis": analysis_result,
            "validated_findings": validated_result,
            "sources": gathered_info.get("sources", []),
            "timestamp": datetime.now().isoformat()
        }
        
        # 将研究结果添加到内存中
        await self.memory_manager.add_to_external_memory(
            f"Research on '{query}': {analysis_result.get('summary', '')}",
            {"type": "research_result", "query": query}
        )
        
        logger.info(f"查询研究完成: {query[:50]}...")
        return result
    
    async def _plan_research(self, query: str, depth: str) -> Dict[str, Any]:
        """根据查询和深度要求规划研究方法"""
        # 根据深度确定研究步骤
        if depth == "basic":
            steps = [
                "执行快速搜索以获取概览",
                "识别关键来源",
                "总结主要观点"
            ]
        elif depth == "standard":
            steps = [
                "执行全面搜索",
                "识别不同领域的多个来源",
                "比较和对比信息",
                "总结发现"
            ]
        else:  # comprehensive
            steps = [
                "执行多方面搜索",
                "识别权威来源",
                "查找原始资料",
                "交叉验证信息",
                "识别潜在矛盾",
                "综合全面概述"
            ]
        
        plan = {
            "query": query,
            "depth": depth,
            "steps": steps,
            "planned_at": datetime.now().isoformat()
        }
        
        # 将计划添加到内存
        await self.memory_manager.add_to_scratchpad(f"Research plan for '{query}': {str(steps)}", {"type": "research_plan"})
        
        return plan
    
    async def _execute_research_plan(self, plan: Dict[str, Any], max_sources: int) -> Dict[str, Any]:
        """执行研究计划以收集信息"""
        query = plan["query"]
        sources = []
        information = []
        
        # 使用网络搜索工具收集初始来源
        search_results = await self.web_search_tool.search(query, max_results=max_sources)
        
        # 处理每个搜索结果
        for i, result in enumerate(search_results):
            if i >= max_sources:
                break
                
            # 将来源添加到我们的来源列表中
            source = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "summary": result.get("content", ""),
                "relevance_score": result.get("relevance_score", 0.0)
            }
            sources.append(source)
            
            # 如果URL可用，则抓取内容
            if source["url"]:
                try:
                    content = await self.scraper_tool.scrape_url(source["url"])
                    information.append({
                        "source": source,
                        "content": content,
                        "extracted_at": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"无法抓取 {source['url']}: {str(e)}")
                    # 如果抓取失败，则使用摘要
                    information.append({
                        "source": source,
                        "content": source["summary"],
                        "extracted_at": datetime.now().isoformat()
                    })
        
        result = {
            "sources": sources,
            "information": information,
            "executed_at": datetime.now().isoformat()
        }
        
        # 将收集的信息添加到内存中
        for info in information:
            await self.memory_manager.add_to_external_memory(
                info["content"],
                {"type": "research_source", "source_url": info["source"].get("url")}
            )
        
        return result
    
    async def _analyze_information(self, gathered_info: Dict[str, Any], query: str) -> Dict[str, Any]:
        """分析收集的信息以提取见解"""
        # 为分析合并所有信息
        all_content = []
        for info in gathered_info.get("information", []):
            all_content.append(info["content"])
        
        combined_content = "\n\n".join(all_content)
        
        # 使用分析代理执行分析
        analysis_prompt = f"""
        分析以下与查询相关的信息: "{query}"
        
        信息:
        {combined_content[:3000]}  # 限制以防止超出令牌限制
        
        提供:
        1. 关键发现
        2. 主要主题或模式
        3. 来源中的矛盾或分歧（如果有的话）
        4. 信息空白
        5. 最重要观点的总结
        """
        
        try:
            analysis_response = await self.analysis_agent.ainvoke([HumanMessage(content=analysis_prompt)])
            analysis_text = analysis_response.content if hasattr(analysis_response, 'content') else str(analysis_response)
        except Exception as e:
            logger.error(f"分析过程中出错: {str(e)}")
            analysis_text = f"分析失败: {str(e)}"
        
        analysis_result = {
            "summary": analysis_text,
            "key_findings": self._extract_key_points(analysis_text),
            "themes": self._extract_themes(analysis_text),
            "contradictions": self._extract_contradictions(analysis_text),
            "gaps": self._extract_gaps(analysis_text),
            "analyzed_at": datetime.now().isoformat()
        }
        
        # 将分析添加到内存中
        await self.memory_manager.add_to_scratchpad(
            f"Analysis for '{query}': {analysis_text[:200]}...",
            {"type": "analysis_result", "query": query}
        )
        
        return analysis_result
    
    async def _validate_findings(self, analysis_result: Dict[str, Any], gathered_info: Dict[str, Any]) -> Dict[str, Any]:
        """验证结果的准确性和可靠性"""
        # 目前，我们实现基本验证
        # 在真实系统中，这将涉及交叉引用、事实核查等
        
        sources = gathered_info.get("sources", [])
        validation_result = {
            "source_credibility_assessment": self._assess_source_credibility(sources),
            "fact_check_status": "pending",  # 将通过事实核查工具实现
            "confidence_level": self._calculate_confidence(analysis_result, sources),
            "validation_notes": "执行基本验证。完整事实核查需要额外工具。",
            "validated_at": datetime.now().isoformat()
        }
        
        return validation_result
    
    def _extract_key_points(self, analysis_text: str) -> List[str]:
        """从分析文本中提取关键点"""
        # 简单提取 - 实际应用中会使用更复杂的NLP
        lines = analysis_text.split('\n')
        key_points = []
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                key_points.append(line)
        return key_points
    
    def _extract_themes(self, analysis_text: str) -> List[str]:
        """从分析文本中提取主题"""
        # 简单提取
        if "themes" in analysis_text.lower():
            # 查找主题部分
            lines = analysis_text.lower().split('\n')
            for line in lines:
                if "themes" in line or "main themes" in line:
                    # 提取主题标题后的内容
                    continue
        return ["需要实现主题识别"]
    
    def _extract_contradictions(self, analysis_text: str) -> List[str]:
        """从分析文本中提取矛盾"""
        # 简单提取
        if "contradictions" in analysis_text.lower():
            # 查找矛盾部分
            pass
        return ["需要实现矛盾识别"]
    
    def _extract_gaps(self, analysis_text: str) -> List[str]:
        """从分析文本中提取信息空白"""
        # 简单提取
        if "gaps" in analysis_text.lower():
            # 查找空白部分
            pass
        return ["需要实现空白识别"]
    
    def _assess_source_credibility(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估来源可信度"""
        # 基于域名声誉的简单可信度评估
        # 实际应用中会使用更复杂的方法
        credible_sources = []
        questionable_sources = []
        
        for source in sources:
            url = source.get("url", "")
            # 基本可信度指标
            is_academic = any(domain in url for domain in ['.edu', '.gov', '.org', 'research', 'journal'])
            is_news = any(domain in url for domain in ['.com', 'news', 'times'])
            
            if is_academic:
                credible_sources.append(source)
            elif is_news:
                questionable_sources.append(source)
            else:
                questionable_sources.append(source)
        
        return {
            "credible_sources_count": len(credible_sources),
            "questionable_sources_count": len(questionable_sources),
            "total_sources": len(sources),
            "assessment_method": "basic_domain_analysis"
        }
    
    def _calculate_confidence(self, analysis_result: Dict[str, Any], sources: List[Dict[str, Any]]) -> float:
        """计算分析的置信度"""
        # 简单置信度计算
        # 实际应用中会更复杂
        source_credibility = len([s for s in sources if any(domain in s.get('url', '') for domain in ['.edu', '.gov', '.org'])])
        total_sources = len(sources)
        
        if total_sources > 0:
            credibility_ratio = source_credibility / total_sources
            # 基于来源可信度(0.5)、来源数量(0.3)和分析质量(0.2)的置信度
            confidence = 0.5 * credibility_ratio + 0.3 * min(1.0, total_sources / 10) + 0.2 * 0.8  # 0.8表示良好分析质量
        else:
            confidence = 0.3  # 如果没有来源则置信度较低
        
        return min(1.0, confidence)  # 限制为1.0
    
    async def generate_report(self, research_result: Dict[str, Any], format_type: str = "markdown", include_sources: bool = True) -> str:
        """根据研究结果生成最终报告"""
        return await self.report_generator.generate_report(
            research_result=research_result,
            format_type=format_type,
            include_sources=include_sources
        )
    
    async def debate_analysis(self, initial_analysis: str, alternative_view: str) -> str:
        """
        模拟多代理辩论以增强分析
        实现架构中的多代理辩论机制
        """
        debate_prompt = f"""
        初始分析: {initial_analysis}
        
        替代观点: {alternative_view}
        
        在这两个观点之间进行结构化辩论。
        1. 识别每个立场的优势
        2. 识别每个立场的劣势
        3. 综合更全面的理解
        4. 识别同意和不同意的领域
        """
        
        try:
            debate_response = await self.llm.ainvoke([HumanMessage(content=debate_prompt)])
            return debate_response.content
        except Exception as e:
            logger.error(f"辩论分析过程中出错: {str(e)}")
            return f"辩论分析失败: {str(e)}"