"""
Research Agent for Deep Research Report Generation Agent System
Implements the reasoning layer based on multi-agent collaboration and dynamic planning
"""
import asyncio
import logging
import requests
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from config import (
    LLM_PROVIDER, 
    OPENAI_BASE_MODEL, 
    OPENAI_TEMPERATURE, 
    OPENAI_MAX_TOKENS,
    ARK_API_KEY,
    DOUBAO_MODEL,
    DOUBAO_API_ENDPOINT,
    DOUBAO_MAX_COMPLETION_TOKENS,
    DOUBAO_REASONING_EFFORT,
    DOUBAO_TEMPERATURE
)
from src.memory.memory_manager import MemoryManager
from src.generation.report_generator import ReportGenerator
from src.tools.web_search_tool import WebSearchTool
from src.tools.scraper_tool import ScraperTool

logger = logging.getLogger(__name__)


class DoubaoLLM:
    """
    豆包LLM客户端，实现与豆包API的交互
    """
    
    def __init__(self, api_key, model, api_endpoint, max_tokens, reasoning_effort, temperature):
        self.api_key = api_key
        self.model = model
        self.api_endpoint = api_endpoint
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        
    def _convert_messages_to_doubao_format(self, messages):
        """将LangChain消息格式转换为豆包API所需格式"""
        doubao_messages = []
        for message in messages:
            # 处理HumanMessage
            if isinstance(message, HumanMessage):
                content = []
                # 检查是否为纯文本消息
                if hasattr(message, 'content') and isinstance(message.content, str):
                    content.append({
                        "text": message.content,
                        "type": "text"
                    })
                # 处理图像内容（如果有）
                elif hasattr(message, 'content') and isinstance(message.content, list):
                    content = message.content
                
                doubao_messages.append({
                    "role": "user",
                    "content": content
                })
            # 处理SystemMessage
            elif isinstance(message, SystemMessage):
                doubao_messages.append({
                    "role": "system",
                    "content": [{
                        "text": message.content,
                        "type": "text"
                    }]
                })
            # 处理AIMessage
            elif isinstance(message, AIMessage):
                doubao_messages.append({
                    "role": "assistant",
                    "content": [{
                        "text": message.content if hasattr(message, 'content') else str(message),
                        "type": "text"
                    }]
                })
        return doubao_messages
    
    async def ainvoke(self, messages):
        """
        异步调用豆包API，包含重试机制
        """
        # 转换消息格式
        doubao_messages = self._convert_messages_to_doubao_format(messages)
        
        # 构建请求体
        payload = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": doubao_messages,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature
        }
        
        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 配置重试参数
        max_retries = 3
        retry_delay = 2  # 初始延迟时间（秒）
        
        for attempt in range(max_retries):
            try:
                logger.info(f"豆包API调用尝试 #{attempt+1}/{max_retries} 到 {self.api_endpoint}")
                
                # 发送请求，优化超时设置
                response = requests.post(
                    self.api_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=(10, 90)  # (连接超时, 读取超时)
                )
                
                # 检查响应状态
                response.raise_for_status()
                
                # 解析响应
                result = response.json()
                
                # 提取内容
                if "choices" in result and result["choices"]:
                    content = result["choices"][0]["message"]["content"]
                    logger.info(f"豆包API调用成功，返回内容长度: {len(content)} 字符")
                    # 创建一个类似AIMessage的对象返回
                    return type('obj', (object,), {"content": content})
                else:
                    error_msg = f"豆包API返回无效响应: {result}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except requests.exceptions.Timeout as e:
                error_msg = f"豆包API请求超时: {str(e)}"
                logger.warning(f"{error_msg} (尝试 {attempt+1}/{max_retries})")
                
                # 如果是最后一次尝试，则抛出异常
                if attempt == max_retries - 1:
                    logger.error(f"豆包API请求在{max_retries}次尝试后全部超时")
                    raise
                
                # 指数退避延迟
                delay = retry_delay * (2 ** attempt)
                logger.info(f"{delay:.2f}秒后重试...")
                await asyncio.sleep(delay)
                
            except requests.exceptions.RequestException as e:
                error_msg = f"豆包API请求失败: {str(e)}"
                logger.error(error_msg)
                raise
            except Exception as e:
                error_msg = f"豆包API调用出错: {str(e)}"
                logger.error(error_msg)
                raise


class ResearchAgent:
    """
    主要研究代理，协调整个研究过程
    基于多代理协作实现推理层
    """
    
    def __init__(self, memory_manager: MemoryManager, report_generator: ReportGenerator):
        # 初始化内存管理器和报告生成器
        self.memory_manager = memory_manager
        self.report_generator = report_generator
        
        # 根据配置选择并初始化大语言模型
        if LLM_PROVIDER == "doubao":
            if not ARK_API_KEY:
                raise ValueError("使用豆包模型时，必须设置ARK_API_KEY环境变量")
            
            logger.info(f"初始化豆包LLM: {DOUBAO_MODEL}")
            self.llm = DoubaoLLM(
                api_key=ARK_API_KEY,
                model=DOUBAO_MODEL,
                api_endpoint=DOUBAO_API_ENDPOINT,
                max_tokens=DOUBAO_MAX_COMPLETION_TOKENS,
                reasoning_effort=DOUBAO_REASONING_EFFORT,
                temperature=DOUBAO_TEMPERATURE
            )
        else:
            # 默认使用OpenAI
            logger.info(f"初始化OpenAI LLM: {OPENAI_BASE_MODEL}")
            self.llm = ChatOpenAI(
                model_name=OPENAI_BASE_MODEL,
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS
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
        if LLM_PROVIDER == "doubao":
            # 对于豆包，直接使用自定义的提示处理·
            system_prompt = "You are a research planning specialist. Your role is to break down complex research queries into manageable steps and create a research plan."
            return type('obj', (object,), {
                'ainvoke': lambda messages: self._doubao_agent_invoke(system_prompt, messages)
            })
        else:
            # 对于OpenAI，使用LangChain的标准方式
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a research planning specialist. Your role is to break down complex research queries into manageable steps and create a research plan."),
                MessagesPlaceholder(variable_name="messages")
            ])
            return prompt | self.llm
    
    def _create_research_agent(self):
        """创建研究代理组件"""
        if LLM_PROVIDER == "doubao":
            system_prompt = "You are a research specialist. Your role is to gather relevant information from various sources based on the research plan."
            return type('obj', (object,), {
                'ainvoke': lambda messages: self._doubao_agent_invoke(system_prompt, messages)
            })
        else:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a research specialist. Your role is to gather relevant information from various sources based on the research plan."),
                MessagesPlaceholder(variable_name="messages")
            ])
            return prompt | self.llm
    
    def _create_analysis_agent(self):
        """创建分析代理组件"""
        if LLM_PROVIDER == "doubao":
            system_prompt = "You are an analysis specialist. Your role is to synthesize gathered information, identify patterns, and draw insights."
            return type('obj', (object,), {
                'ainvoke': lambda messages: self._doubao_agent_invoke(system_prompt, messages)
            })
        else:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are an analysis specialist. Your role is to synthesize gathered information, identify patterns, and draw insights."),
                MessagesPlaceholder(variable_name="messages")
            ])
            return prompt | self.llm
    
    def _create_validation_agent(self):
        """创建验证代理组件"""
        if LLM_PROVIDER == "doubao":
            system_prompt = "You are a validation specialist. Your role is to verify the accuracy and reliability of information and conclusions."
            return type('obj', (object,), {
                'ainvoke': lambda messages: self._doubao_agent_invoke(system_prompt, messages)
            })
        else:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a validation specialist. Your role is to verify the accuracy and reliability of information and conclusions."),
                MessagesPlaceholder(variable_name="messages")
            ])
            return prompt | self.llm
    
    async def _doubao_agent_invoke(self, system_prompt, messages):
        """豆包模型的代理调用辅助方法"""
        # 构建完整消息列表，包含系统消息
        full_messages = [SystemMessage(content=system_prompt)] + messages
        # 调用豆包LLM
        return await self.llm.ainvoke(full_messages)
    
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
        
        subtasks = await self._decompose_intent(query)
        self._log_stage("intent_analysis", {"query": query, "subtasks": subtasks})
        gathered_info = await self._execute_subtasks(subtasks, max_sources)
        self._log_stage("subtasks_executed", {"source_count": len(gathered_info.get("sources", [])), "info_count": len(gathered_info.get("information", []))})
        
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

    def _log_stage(self, stage: str, data: Dict[str, Any]):
        logger.info(f"[{datetime.now().isoformat()}] stage.{stage} {data}")

    async def _decompose_intent(self, query: str) -> List[Dict[str, Any]]:
        base_topics = ["背景", "关键技术", "应用场景", "优劣分析", "发展趋势"]
        tasks = []
        for t in base_topics:
            tasks.append({"topic": t, "query": f"{query} {t}", "keywords": [query, t]})
        return tasks

    async def _execute_subtasks(self, subtasks: List[Dict[str, Any]], max_sources: int) -> Dict[str, Any]:
        sources: List[Dict[str, Any]] = []
        information: List[Dict[str, Any]] = []
        for st in subtasks:
            q = st.get("query", "")
            self._log_stage("query_optimization", {"subtopic": st.get("topic"), "query": q})
            results = await self.web_search_tool.search(q, max_results=max_sources)
            self._log_stage("subtask_search", {"subtopic": st.get("topic"), "result_count": len(results)})
            for i, r in enumerate(results[:max_sources]):
                src = {"title": r.get("title", ""), "url": r.get("url", ""), "summary": r.get("content", ""), "relevance_score": r.get("relevance_score", 0.0), "subtopic": st.get("topic")}
                sources.append(src)
                if src["url"]:
                    try:
                        content = await self.scraper_tool.scrape_url(src["url"])
                    except Exception:
                        content = src["summary"]
                else:
                    content = src["summary"]
                information.append({"source": src, "content": content, "extracted_at": datetime.now().isoformat(), "subtopic": st.get("topic")})
                await self.memory_manager.add_to_external_memory(content, {"type": "research_source", "source_url": src.get("url"), "subtopic": st.get("topic")})
            sub_infos = [inf for inf in information if inf.get("subtopic") == st.get("topic")]
            joined = "\n\n".join([inf["content"][:2000] for inf in sub_infos])
            summary = joined[:800]
            self._log_stage("segmented_processing", {"subtopic": st.get("topic"), "segments": len(sub_infos), "summary_len": len(summary)})
            await self.memory_manager.add_to_scratchpad(summary, {"type": "subtopic_summary", "subtopic": st.get("topic")})
        return {"sources": sources, "information": information, "executed_at": datetime.now().isoformat()}
    
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
    
    def _extract_and_expand_keywords(self, query: str) -> str:
        """从查询中提取和扩展关键词以增强搜索效果"""
        # 去除多余空格并确保查询清晰
        query = ' '.join(query.split())
        
        # 这里可以添加更复杂的关键词提取逻辑
        # 例如使用NLP工具提取关键实体、名词短语等
        
        logger.info(f"处理查询关键词: {query}")
        return query
    
    async def _execute_research_plan(self, plan: Dict[str, Any], max_sources: int) -> Dict[str, Any]:
        """执行研究计划以收集信息"""
        query = plan["query"]
        
        # 提取和扩展关键词
        processed_query = self._extract_and_expand_keywords(query)
        
        sources = []
        information = []
        
        # 使用处理后的查询进行网络搜索
        logger.info(f"使用处理后的查询进行搜索: {processed_query}")
        search_results = await self.web_search_tool.search(processed_query, max_results=max_sources)
        
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
