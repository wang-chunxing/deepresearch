"""
Research Agent for Deep Research Report Generation Agent System
Implements the reasoning layer based on multi-agent collaboration and dynamic planning
"""
import asyncio
import logging
import requests
import os
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None
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
from src.tools.fact_check_tool import FactCheckTool

logger = logging.getLogger(__name__)


from src.llm.doubao import DoubaoLLM


class ResearchAgent:
    """
    主要研究代理，协调整个研究过程
    基于多代理协作实现推理层
    """
    
    def __init__(self, memory_manager: MemoryManager, report_generator: ReportGenerator, learning_manager=None):
        # 初始化内存管理器和报告生成器
        self.memory_manager = memory_manager
        self.report_generator = report_generator
        self.learning_manager = learning_manager
        
        # 根据配置选择并初始化大语言模型
        if LLM_PROVIDER == "doubao":
            if not ARK_API_KEY:
                logger.warning("ARK_API_KEY 未设置，使用测试LLM占位符")
                class _DummyLLM:
                    async def ainvoke(self, messages):
                        return AIMessage(content="")
                self.llm = _DummyLLM()
                self._use_openai_pipeline = False
            else:
                logger.info(f"初始化豆包LLM: {DOUBAO_MODEL}")
                self.llm = DoubaoLLM(
                    api_key=ARK_API_KEY,
                    model=DOUBAO_MODEL,
                    api_endpoint=DOUBAO_API_ENDPOINT,
                    max_tokens=DOUBAO_MAX_COMPLETION_TOKENS,
                    reasoning_effort=DOUBAO_REASONING_EFFORT,
                    temperature=DOUBAO_TEMPERATURE
                )
                self._use_openai_pipeline = False
        else:
            # 默认使用OpenAI
            if ChatOpenAI is None or not os.getenv("OPENAI_API_KEY", ""):
                logger.warning("OPENAI_API_KEY 未设置，使用测试LLM占位符")
                class _DummyLLM:
                    async def ainvoke(self, messages):
                        return AIMessage(content="")
                self.llm = _DummyLLM()
                self._use_openai_pipeline = False
            else:
                logger.info(f"初始化OpenAI LLM: {OPENAI_BASE_MODEL}")
                self.llm = ChatOpenAI(
                    model_name=OPENAI_BASE_MODEL,
                    temperature=OPENAI_TEMPERATURE,
                    max_tokens=OPENAI_MAX_TOKENS
                )
                self._use_openai_pipeline = True
        
        # 初始化工具
        self.web_search_tool = WebSearchTool()
        self.scraper_tool = ScraperTool()
        self.fact_check_tool = FactCheckTool()
        
        # 初始化专门代理（通过不同提示/角色模拟）
        self.planning_agent = self._create_planning_agent()
        self.research_agent = self._create_research_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.validation_agent = self._create_validation_agent()
        
        logger.info("研究代理已使用专门代理组件初始化")
    
    def _create_planning_agent(self):
        """创建规划代理组件"""
        if not getattr(self, "_use_openai_pipeline", False):
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
        if not getattr(self, "_use_openai_pipeline", False):
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
        if not getattr(self, "_use_openai_pipeline", False):
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
        if not getattr(self, "_use_openai_pipeline", False):
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

    def _parse_input(self, query: str) -> Dict[str, Any]:
        normalized = ' '.join(query.split())
        return {"normalized_query": normalized, "entities": [], "relations": [], "dimensions": []}

    async def _decompose_intent(self, query: str) -> List[Dict[str, Any]]:
        base_topics = ["背景", "关键技术", "应用场景", "优劣分析", "发展趋势"]
        tasks = []
        for t in base_topics:
            tasks.append({"topic": t, "query": f"{query} {t}", "keywords": [query, t]})
        return tasks

    async def _infer_parameters(self, parsed: Dict[str, Any], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        q = parsed.get("normalized_query", "")
        length = len(q)
        if length < 40:
            depth = "basic"
            breadth = 5
            iterations = 3
        elif length < 120:
            depth = "standard"
            breadth = 6
            iterations = 3
        else:
            depth = "comprehensive"
            breadth = 8
            iterations = 4
        return {"strategy": {"breadth": breadth, "depth": 2 if depth != "basic" else 1, "iterations": iterations}, "analysis_depth": depth, "report_detail": "standard"}

    async def _execute_subtasks(self, subtasks: List[Dict[str, Any]], max_sources: int, parallel: bool = False) -> Dict[str, Any]:
        sources: List[Dict[str, Any]] = []
        information: List[Dict[str, Any]] = []
        trace: List[Dict[str, Any]] = []
        async def handle_subtask(st: Dict[str, Any]):
            q = st.get("query", "")
            self._log_stage("query_optimization", {"subtopic": st.get("topic"), "query": q})
            results = await self.web_search_tool.search(q, max_results=max_sources)
            self._log_stage("subtask_search", {"subtopic": st.get("topic"), "result_count": len(results)})
            trace.append({"topic": st.get("topic"), "query": q, "sources": [r.get("url") for r in results]})
            for i, r in enumerate(results[:max_sources]):
                src = {"title": r.get("title", ""), "url": r.get("url", ""), "summary": r.get("content", ""), "relevance_score": r.get("relevance_score", 0.0), "subtopic": st.get("topic")}
                sources.append(src)
                if src["url"]:
                    try:
                        if hasattr(self.scraper_tool, 'should_scrape') and not self.scraper_tool.should_scrape(src["url"]):
                            content = src["summary"]
                        else:
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
        if parallel:
            await asyncio.gather(*(handle_subtask(st) for st in subtasks))
        else:
            for st in subtasks:
                await handle_subtask(st)
        return {"sources": sources, "information": information, "trace": trace, "executed_at": datetime.now().isoformat()}
    
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

    async def _initial_bfs_collect(self, parsed: Dict[str, Any], breadth: int) -> Dict[str, Any]:
        base = parsed.get("normalized_query", "")
        variants = self._generate_query_variants(base, [], breadth)
        subtasks = [{"topic": v.get("topic"), "query": v.get("query"), "keywords": v.get("keywords", [])} for v in variants[:breadth]]
        return await self._execute_subtasks(subtasks, max_sources=breadth, parallel=True)

    def _analyze_gaps(self, initial_results: Dict[str, Any]) -> Dict[str, Any]:
        infos = initial_results.get("information", [])
        topics = list({i.get("subtopic") for i in infos if i.get("subtopic")})
        domain_terms = ["供应链", "物流", "制造", "库存", "需求预测", "调度", "质量"]
        seen_text = " ".join([i.get("content", "")[:2000] for i in infos])
        missing = [t for t in ["数据", "方法", "案例", "评价", "行业报告", "论文"] if t not in topics or t not in seen_text]
        conflicts = []
        return {"topics": topics, "missing": missing, "conflicts": conflicts}

    async def _targeted_followups(self, parsed: Dict[str, Any], gaps: Dict[str, Any], depth: int) -> Dict[str, Any]:
        base = parsed.get("normalized_query", "")
        targets = gaps.get("missing", [])[:max(1, depth)]
        variants = self._generate_query_variants(base, targets, max(3, depth))
        subtasks = [{"topic": v.get("topic"), "query": v.get("query"), "keywords": v.get("keywords", [])} for v in variants]
        return await self._execute_subtasks(subtasks, max_sources=5, parallel=True)

    def _collect_sources(self, initial_results: Dict[str, Any], followup_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        s = list(initial_results.get("sources", []))
        for fr in followup_results:
            s.extend(fr.get("sources", []))
        for x in s:
            url = x.get("url", "")
            domain = url.split("/")[2] if url and "/" in url else ""
            x["domain"] = domain
        return s

    def _generate_query_variants(self, base: str, targets: List[str], count: int) -> List[Dict[str, Any]]:
        lw = self.learning_manager.get_synonym_weights() if self.learning_manager else {}
        synonyms = self._domain_synonyms(base)
        synonyms.sort(key=lambda s: lw.get(s, 0.0), reverse=True)
        years = ["2024", "2025"]
        academic_filters = ["site:arxiv.org", "site:sciencedirect.com", "site:mdpi.com", "site:springer.com"]
        industry_terms = ["行业报告", "白皮书", "案例研究", "实践效果", "落地挑战", "前沿趋势"]
        variants: List[Dict[str, Any]] = []
        base_core = base
        seeds = targets if targets else ["背景", "关键技术", "应用场景", "挑战", "趋势"]
        for i, t in enumerate(seeds):
            syn = random.choice(synonyms) if synonyms else "研究"
            q = f"{base_core} {t} {syn}"
            if i % 2 == 0:
                q = f"{q} {random.choice(years)}"
            if i % 3 == 0:
                q = f"{q} {random.choice(academic_filters)}"
            variants.append({"topic": t, "query": q, "keywords": [base_core, t]})
        for f in industry_terms[:max(1, count - len(variants))]:
            syn = random.choice(synonyms) if synonyms else "研究"
            q = f"{base_core} {f} {syn}"
            variants.append({"topic": f, "query": q, "keywords": [base_core, f]})
        return variants[:count]

    def _domain_synonyms(self, base: str) -> List[str]:
        b = base.lower()
        if any(k in b for k in ["供应链", "logistics", "制造", "manufacturing"]):
            return ["供应链", "物流", "制造", "库存优化", "需求预测", "动态调度", "质量检测"]
        if any(k in b for k in ["金融", "风控", "finance", "risk"]):
            return ["金融风控", "信用评估", "反欺诈", "合规", "模型治理", "实证研究"]
        return ["综述", "实践", "案例", "方法", "评估", "趋势"]

    def _generate_outline(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        q = parsed.get("normalized_query", "")
        directions = [
            {"title": "背景与问题界定", "keywords": [q, "背景", "现状"], "path": ["权威百科", "行业概览", "政策文件"]},
            {"title": "关键技术与方法", "keywords": [q, "关键技术", "方法"], "path": ["学术论文", "技术白皮书", "开源实现"]},
            {"title": "应用场景与案例", "keywords": [q, "应用场景", "案例"], "path": ["行业报告", "案例分析", "企业实践"]},
            {"title": "优势劣势与挑战", "keywords": [q, "优劣分析", "挑战"], "path": ["对比研究", "失败案例", "风险合规"]},
            {"title": "趋势与建议", "keywords": [q, "前沿趋势", "建议"], "path": ["趋势报告", "投研分析", "专家意见"]},
        ]
        return directions[:5]

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
        
        sources = gathered_info.get("sources", [])
        claims = []
        for kp in analysis_result.get("key_findings", [])[:5]:
            claims.append(str(kp))
        fc = await self._fact_check(claims, sources)
        validation_result = {
            "source_credibility_assessment": self._assess_source_credibility(sources),
            "fact_check_status": fc.get("status"),
            "fact_check_score": fc.get("score"),
            "fact_check_details": fc.get("details", []),
            "confidence_level": self._calculate_confidence(analysis_result, sources),
            "validation_notes": "包含基础事实核查与来源可信度评估",
            "validated_at": datetime.now().isoformat()
        }
        
        return validation_result

    async def _fact_check(self, claims: List[str], context_sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self.fact_check_tool.check_claims(claims, context_sources)
    
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

    async def _compose_sections_v2(self, gathered_info: Dict[str, Any], outline: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        infos = gathered_info.get("information", [])
        top_infos = infos[:6]
        evidence_items = []
        for idx, info in enumerate(top_infos):
            src = info.get("source", {})
            evidence_items.append({
                "id": idx + 1,
                "title": src.get("title", ""),
                "url": src.get("url", ""),
                "snippet": (info.get("content", "")[:800] if info.get("content") else src.get("summary", "")[:800])
            })
        corpus_blocks = []
        for ev in evidence_items:
            corpus_blocks.append(f"[#{ev['id']}] {ev['title']}\n{ev['snippet']}")
        corpus = "\n\n".join(corpus_blocks)
        priorities = [d.get("title", "") for d in (outline or [])][:5]
        prompt = (
            "你是一名专业中文研究报告撰写专家。\n"
            f"命题：{query}\n"
            f"优先结构方向：{', '.join(priorities)}\n"
            "请严格基于下列真实证据（每条以[#编号]标注），输出JSON：\n"
            "字段：findings(数组:{text, evidence_ids}), conclusions(数组:{text, evidence_ids}), recommendations(数组:{text, reason, evidence_ids})。\n"
            "要求：内容必须为中文、具体、避免套话；每部分3-6条；发现/结论均需引用证据编号；建议需给出依据。\n"
            "证据如下：\n" + corpus
        )
        findings: List[Dict[str, Any]] = []
        conclusions: List[Dict[str, Any]] = []
        recommendations: List[Dict[str, Any]] = []
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            txt = resp.content if hasattr(resp, "content") else str(resp)
            import json, re
            m = re.search(r"\{[\s\S]*\}$", txt.strip())
            payload = json.loads(m.group(0) if m else txt)
            findings = payload.get("findings", [])
            conclusions = payload.get("conclusions", [])
            recommendations = payload.get("recommendations", [])
        except Exception:
            findings = [{"text": "需补充领域数据以识别关键瓶颈", "evidence_ids": []}]
            conclusions = [{"text": "现有证据不足以形成强结论", "evidence_ids": []}]
            recommendations = [{"text": "追加检索、补充权威来源并开展试点验证", "reason": "提高证据充分性", "evidence_ids": []}]
        return {
            "findings": findings[:6],
            "conclusions": conclusions[:6],
            "recommendations": recommendations[:6],
            "evidence": evidence_items
        }
    async def _compose_longform_article(self, module: Dict[str, Any], query: str, outline: List[Dict[str, Any]]) -> str:
        priorities = [d.get("title", "") for d in (outline or [])][:5]
        findings = module.get("findings", [])
        conclusions = module.get("conclusions", [])
        recommendations = module.get("recommendations", [])
        evidence = module.get("evidence", [])
        corpus_blocks = []
        for ev in evidence[:10]:
            corpus_blocks.append(f"[#{ev.get('id')}] {ev.get('title')}\n{ev.get('snippet','')}")
        corpus = "\n\n".join(corpus_blocks)
        prompt = (
            "你是一位专业中文研报撰写专家，请据以下命题与证据撰写一篇完整文章（非提纲、非要点）。\n"
            f"命题：{query}\n"
            f"结构方向：{', '.join(priorities)}\n"
            "写作要求：\n- 文章体，段落化，避免列表与套话；\n- 严谨、可读性强，必要处引用证据编号[#n]；\n- 结尾给出可执行建议（不超过3条）。\n"
            "证据：\n" + corpus + "\n"
            "可参考要点：\n" + "\n".join([f"- 发现：{f.get('text','')}" for f in findings[:5]]) + "\n"
            + "\n".join([f"- 结论：{c.get('text','')}" for c in conclusions[:5]])
        )
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            # 本地回退：基于模块要点合成段落文章
            parts = []
            parts.append(f"围绕命题“{query}”，我们根据公开来源开展系统性研究，优先参考{', '.join([d for d in priorities if d])}等方向。")
            if findings:
                pf = "；".join([f.get("text", "") for f in findings[:5] if f.get("text")])
                parts.append(f"核心发现包括：{pf}。")
            if conclusions:
                pc = "；".join([c.get("text", "") for c in conclusions[:5] if c.get("text")])
                parts.append(f"据此，我们形成的分析结论为：{pc}。")
            if recommendations:
                pr = "；".join([r.get("text", "") for r in recommendations[:3] if r.get("text")])
                parts.append(f"基于上述证据与分析，建议：{pr}。")
            if evidence:
                refs = ", ".join([f"[#${ev.get('id')}]" for ev in evidence[:6]])
                parts.append(f"文中关键论点可对应以上证据编号（例如：{refs}）。")
            return "\n\n".join(parts)

    async def _polish_article(self, raw_article: str, module: Dict[str, Any], query: str) -> str:
        findings = module.get("findings", [])
        conclusions = module.get("conclusions", [])
        evidence = module.get("evidence", [])
        refs = []
        for ev in evidence[:10]:
            t = ev.get("title", "")
            u = ev.get("url", "")
            if t or u:
                refs.append(f"- {t} {u}")
        prompt = (
            "请将以下研究内容润色为一篇可直接面向读者发布的中文报告："
            f"\n题目：{query}"
            "\n要求：段落化叙述，去除列表化与JSON痕迹，语言专业但易读，逻辑清晰，避免模板语；适度引用证据编号；结尾附上简短建议与参考来源。"
            "\n原始正文：\n" + (raw_article or "") + "\n"
            "\n关键发现：\n" + "\n".join([f.get("text", "") for f in findings[:6]]) + "\n"
            "\n分析结论：\n" + "\n".join([c.get("text", "") for c in conclusions[:6]]) + "\n"
            "\n参考来源（标题与链接，仅供写作参考）：\n" + "\n".join(refs)
        )
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return resp.content if hasattr(resp, "content") else str(resp)
        except Exception:
            return raw_article
    async def _compose_sections(self, gathered_info: Dict[str, Any], outline: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        infos = gathered_info.get("information", [])
        # 选取最相关的前6条信息作为证据
        top_infos = infos[:6]
        evidence_items = []
        for idx, info in enumerate(top_infos):
            src = info.get("source", {})
            evidence_items.append({
                "id": idx + 1,
                "title": src.get("title", ""),
                "url": src.get("url", ""),
                "snippet": (info.get("content", "")[:800] if info.get("content") else src.get("summary", "")[:800])
            })
        # 组装上下文（截断以控制长度）
        corpus_blocks = []
        for ev in evidence_items:
            corpus_blocks.append(f"[#{ev['id']}] {ev['title']}\n{ev['snippet']}")
        corpus = "\n\n".join(corpus_blocks)
        # 领域方向用于强调优先结构
        priorities = [d.get("title", "") for d in (outline or [])][:5]
        prompt = (
            "你是一名专业中文研究报告撰写专家。\n"
            f"命题：{query}\n"
            f"优先结构方向：{', '.join(priorities)}\n"
            "请严格基于下列真实证据（每条以[#编号]标注），生成 JSON 格式输出：\n"
            "{\n  findings: [{text: '示例', evidence_ids: [编号]}],\n"
            "  conclusions: [{text: '示例', evidence_ids: [编号]}],\n"
            "  recommendations: [{text: '示例', reason: '依据与可行性', evidence_ids: [编号]}]\n}"
            "要求：\n- 内容必须为中文、具体、避免套话；\n- 每部分3-6条；\n- findings与conclusions需引用相关证据编号；\n- recommendations需可执行、说明依据。\n"
            "证据如下：\n" + corpus
        )
        findings: List[Dict[str, Any]] = []
        conclusions: List[Dict[str, Any]] = []
        recommendations: List[Dict[str, Any]] = []
        try:
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            txt = resp.content if hasattr(resp, "content") else str(resp)
            import json, re
            # 尝试提取JSON块
            m = re.search(r"\{[\s\S]*\}$", txt.strip())
            payload = json.loads(m.group(0) if m else txt)
            findings = payload.get("findings", [])
            conclusions = payload.get("conclusions", [])
            recommendations = payload.get("recommendations", [])
        except Exception:
            findings = [{"text": "需补充领域数据以识别关键瓶颈", "evidence_ids": []}]
            conclusions = [{"text": "现有证据不足以形成强结论", "evidence_ids": []}]
            recommendations = [{"text": "追加检索、补充权威来源并开展试点验证", "reason": "提高证据充分性", "evidence_ids": []}]
        return {
            "findings": findings[:6],
            "conclusions": conclusions[:6],
            "recommendations": recommendations[:6],
            "evidence": evidence_items
        }
