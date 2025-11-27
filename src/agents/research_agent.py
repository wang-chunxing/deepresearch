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
    Main research agent that coordinates the research process
    Implements the reasoning layer based on multi-agent collaboration
    """
    
    def __init__(self, memory_manager: MemoryManager, report_generator: ReportGenerator):
        self.memory_manager = memory_manager
        self.report_generator = report_generator
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model_name=LLM_BASE_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        # Initialize tools
        self.web_search_tool = WebSearchTool()
        self.scraper_tool = ScraperTool()
        
        # Initialize specialized agents (simulated through different prompts/roles)
        self.planning_agent = self._create_planning_agent()
        self.research_agent = self._create_research_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.validation_agent = self._create_validation_agent()
        
        logger.info("Research Agent initialized with specialized agent components")
    
    def _create_planning_agent(self):
        """Create the planning agent component"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a research planning specialist. Your role is to break down complex research queries into manageable steps and create a research plan."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | self.llm
    
    def _create_research_agent(self):
        """Create the research agent component"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a research specialist. Your role is to gather relevant information from various sources based on the research plan."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | self.llm
    
    def _create_analysis_agent(self):
        """Create the analysis agent component"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are an analysis specialist. Your role is to synthesize gathered information, identify patterns, and draw insights."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | self.llm
    
    def _create_validation_agent(self):
        """Create the validation agent component"""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a validation specialist. Your role is to verify the accuracy and reliability of information and conclusions."),
            MessagesPlaceholder(variable_name="messages")
        ])
        return prompt | self.llm
    
    async def research(self, query: str, max_sources: int = 10, depth: str = "comprehensive") -> Dict[str, Any]:
        """
        Conduct research on the given query
        Implements the multi-agent collaboration and reasoning chain building
        """
        logger.info(f"Starting research for query: {query[:50]}...")
        
        # Add query to conversation history
        await self.memory_manager.add_to_conversation_history("user", query)
        
        # Step 1: Plan the research
        research_plan = await self._plan_research(query, depth)
        logger.debug("Research plan created")
        
        # Step 2: Execute research plan
        gathered_info = await self._execute_research_plan(research_plan, max_sources)
        logger.debug(f"Gathered {len(gathered_info.get('sources', []))} sources")
        
        # Step 3: Analyze information
        analysis_result = await self._analyze_information(gathered_info, query)
        logger.debug("Information analysis completed")
        
        # Step 4: Validate findings
        validated_result = await self._validate_findings(analysis_result, gathered_info)
        logger.debug("Findings validation completed")
        
        # Combine results
        result = {
            "query": query,
            "research_plan": research_plan,
            "gathered_information": gathered_info,
            "analysis": analysis_result,
            "validated_findings": validated_result,
            "sources": gathered_info.get("sources", []),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add research result to memory
        await self.memory_manager.add_to_external_memory(
            f"Research on '{query}': {analysis_result.get('summary', '')}",
            {"type": "research_result", "query": query}
        )
        
        logger.info(f"Research completed for query: {query[:50]}...")
        return result
    
    async def _plan_research(self, query: str, depth: str) -> Dict[str, Any]:
        """Plan the research approach based on query and depth requirement"""
        # Determine research steps based on depth
        if depth == "basic":
            steps = [
                "Perform quick search for overview",
                "Identify key sources",
                "Summarize main points"
            ]
        elif depth == "standard":
            steps = [
                "Perform comprehensive search",
                "Identify multiple sources across different domains",
                "Compare and contrast information",
                "Summarize findings"
            ]
        else:  # comprehensive
            steps = [
                "Perform multi-faceted search",
                "Identify authoritative sources",
                "Look for primary sources",
                "Cross-reference information",
                "Identify potential contradictions",
                "Synthesize comprehensive overview"
            ]
        
        plan = {
            "query": query,
            "depth": depth,
            "steps": steps,
            "planned_at": datetime.now().isoformat()
        }
        
        # Add plan to memory
        await self.memory_manager.add_to_scratchpad(f"Research plan for '{query}': {str(steps)}", {"type": "research_plan"})
        
        return plan
    
    async def _execute_research_plan(self, plan: Dict[str, Any], max_sources: int) -> Dict[str, Any]:
        """Execute the research plan to gather information"""
        query = plan["query"]
        sources = []
        information = []
        
        # Use web search tool to gather initial sources
        search_results = await self.web_search_tool.search(query, max_results=max_sources)
        
        # Process each search result
        for i, result in enumerate(search_results):
            if i >= max_sources:
                break
                
            # Add source to our sources list
            source = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "summary": result.get("content", ""),
                "relevance_score": result.get("relevance_score", 0.0)
            }
            sources.append(source)
            
            # Scrape content if URL is available
            if source["url"]:
                try:
                    content = await self.scraper_tool.scrape_url(source["url"])
                    information.append({
                        "source": source,
                        "content": content,
                        "extracted_at": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Failed to scrape {source['url']}: {str(e)}")
                    # Use the summary if scraping fails
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
        
        # Add gathered information to memory
        for info in information:
            await self.memory_manager.add_to_external_memory(
                info["content"],
                {"type": "research_source", "source_url": info["source"].get("url")}
            )
        
        return result
    
    async def _analyze_information(self, gathered_info: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Analyze gathered information to extract insights"""
        # Combine all information for analysis
        all_content = []
        for info in gathered_info.get("information", []):
            all_content.append(info["content"])
        
        combined_content = "\n\n".join(all_content)
        
        # Perform analysis using the analysis agent
        analysis_prompt = f"""
        Analyze the following information related to the query: "{query}"
        
        Information:
        {combined_content[:3000]}  # Limit to prevent exceeding token limits
        
        Provide:
        1. Key findings
        2. Main themes or patterns
        3. Contradictions or disagreements in sources (if any)
        4. Gaps in information
        5. Summary of the most important points
        """
        
        try:
            analysis_response = await self.analysis_agent.ainvoke([HumanMessage(content=analysis_prompt)])
            analysis_text = analysis_response.content if hasattr(analysis_response, 'content') else str(analysis_response)
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            analysis_text = f"Analysis failed: {str(e)}"
        
        analysis_result = {
            "summary": analysis_text,
            "key_findings": self._extract_key_points(analysis_text),
            "themes": self._extract_themes(analysis_text),
            "contradictions": self._extract_contradictions(analysis_text),
            "gaps": self._extract_gaps(analysis_text),
            "analyzed_at": datetime.now().isoformat()
        }
        
        # Add analysis to memory
        await self.memory_manager.add_to_scratchpad(
            f"Analysis for '{query}': {analysis_text[:200]}...",
            {"type": "analysis_result", "query": query}
        )
        
        return analysis_result
    
    async def _validate_findings(self, analysis_result: Dict[str, Any], gathered_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the findings for accuracy and reliability"""
        # For now, we'll implement a basic validation
        # In a real system, this would involve cross-referencing, fact-checking, etc.
        
        sources = gathered_info.get("sources", [])
        validation_result = {
            "source_credibility_assessment": self._assess_source_credibility(sources),
            "fact_check_status": "pending",  # Would be implemented with fact-checking tools
            "confidence_level": self._calculate_confidence(analysis_result, sources),
            "validation_notes": "Basic validation performed. Full fact-checking requires additional tools.",
            "validated_at": datetime.now().isoformat()
        }
        
        return validation_result
    
    def _extract_key_points(self, analysis_text: str) -> List[str]:
        """Extract key points from analysis text"""
        # Simple extraction - in practice, this would use more sophisticated NLP
        lines = analysis_text.split('\n')
        key_points = []
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                key_points.append(line)
        return key_points
    
    def _extract_themes(self, analysis_text: str) -> List[str]:
        """Extract themes from analysis text"""
        # Simple extraction
        if "themes" in analysis_text.lower():
            # Look for themes section
            lines = analysis_text.lower().split('\n')
            for line in lines:
                if "themes" in line or "main themes" in line:
                    # Extract content after the themes heading
                    continue
        return ["Theme identification needs implementation"]
    
    def _extract_contradictions(self, analysis_text: str) -> List[str]:
        """Extract contradictions from analysis text"""
        # Simple extraction
        if "contradictions" in analysis_text.lower():
            # Look for contradictions section
            pass
        return ["Contradiction identification needs implementation"]
    
    def _extract_gaps(self, analysis_text: str) -> List[str]:
        """Extract information gaps from analysis text"""
        # Simple extraction
        if "gaps" in analysis_text.lower():
            # Look for gaps section
            pass
        return ["Gap identification needs implementation"]
    
    def _assess_source_credibility(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess credibility of sources"""
        # Simple credibility assessment based on domain reputation
        # In practice, this would use more sophisticated methods
        credible_sources = []
        questionable_sources = []
        
        for source in sources:
            url = source.get("url", "")
            # Basic credibility indicators
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
        """Calculate confidence level in the analysis"""
        # Simple confidence calculation
        # In practice, this would be more sophisticated
        source_credibility = len([s for s in sources if any(domain in s.get('url', '') for domain in ['.edu', '.gov', '.org'])])
        total_sources = len(sources)
        
        if total_sources > 0:
            credibility_ratio = source_credibility / total_sources
            # Base confidence on source credibility (0.5) and number of sources (0.3) and analysis quality (0.2)
            confidence = 0.5 * credibility_ratio + 0.3 * min(1.0, total_sources / 10) + 0.2 * 0.8  # 0.8 for good analysis quality
        else:
            confidence = 0.3  # Low confidence if no sources
        
        return min(1.0, confidence)  # Cap at 1.0
    
    async def generate_report(self, research_result: Dict[str, Any], format_type: str = "markdown", include_sources: bool = True) -> str:
        """Generate a final report from the research result"""
        return await self.report_generator.generate_report(
            research_result=research_result,
            format_type=format_type,
            include_sources=include_sources
        )
    
    async def debate_analysis(self, initial_analysis: str, alternative_view: str) -> str:
        """
        Simulate multi-agent debate to enhance analysis
        Implements the multi-agent debate mechanism from the architecture
        """
        debate_prompt = f"""
        Initial Analysis: {initial_analysis}
        
        Alternative View: {alternative_view}
        
        Conduct a structured debate between these two viewpoints.
        1. Identify strengths of each position
        2. Identify weaknesses of each position
        3. Synthesize a more comprehensive understanding
        4. Identify areas of agreement and disagreement
        """
        
        try:
            debate_response = await self.llm.ainvoke([HumanMessage(content=debate_prompt)])
            return debate_response.content
        except Exception as e:
            logger.error(f"Error during debate analysis: {str(e)}")
            return f"Debate analysis failed: {str(e)}"