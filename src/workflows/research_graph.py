import asyncio
import time
import hashlib
from typing import Any, Dict, List, Optional

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False


class ResearchGraphRunner:
    def __init__(self, research_agent, memory_manager, report_generator, config, learning_manager=None):
        self.research_agent = research_agent
        self.memory_manager = memory_manager
        self.report_generator = report_generator
        self.config = config
        self.learning_manager = learning_manager
        self._events: List[Dict[str, Any]] = []

    def _hash(self, x: Any) -> str:
        return hashlib.sha256(str(x).encode("utf-8")).hexdigest()[:16]

    def _log_event(self, node: str, inputs: Dict[str, Any], outputs: Dict[str, Any], rationale: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None) -> None:
        data = {
            "event_id": self._hash({"node": node, "ts": time.time(), "inputs": inputs}),
            "node": node,
            "inputs_hash": self._hash(inputs),
            "outputs_hash": self._hash(outputs),
            "rationale": rationale or "",
            "metrics": metrics or {},
            "timestamp": time.time(),
            "task_id": outputs.get("task_id") or inputs.get("task_id")
        }
        self.research_agent._log_stage(f"graph.{node}", data)
        self._events.append(data)

    async def _input_parsing(self, state: Dict[str, Any]) -> Dict[str, Any]:
        parsed = self.research_agent._parse_input(state["query"]) if hasattr(self.research_agent, "_parse_input") else {"normalized_query": state["query"], "entities": [], "relations": [], "dimensions": []}
        out = {**state, "parsed": parsed}
        self._log_event("input_parsing", state, out, "输入解析")
        return out

    async def _task_decomposition(self, state: Dict[str, Any]) -> Dict[str, Any]:
        nq = state["parsed"].get("normalized_query", state["query"]) if isinstance(state.get("parsed"), dict) else state["query"]
        tasks = await self.research_agent._decompose_intent(nq) if hasattr(self.research_agent, "_decompose_intent") else [{"id": "t1", "title": nq, "deps": []}]
        out = {**state, "tasks": tasks}
        self._log_event("task_decomposition", state, out, "任务拆解")
        return out

    async def _outline_generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        outline = self.research_agent._generate_outline(state.get("parsed", {})) if hasattr(self.research_agent, "_generate_outline") else []
        out = {**state, "outline": outline}
        self._log_event("outline_generation", state, out, "生成研究大纲")
        return out

    async def _parameter_inference(self, state: Dict[str, Any]) -> Dict[str, Any]:
        inferred = await self.research_agent._infer_parameters(state.get("parsed", {}), state.get("tasks", [])) if hasattr(self.research_agent, "_infer_parameters") else {"strategy": {"breadth": 6, "depth": 2, "iterations": 3}, "analysis_depth": "standard", "report_detail": "standard"}
        out = {**state, "params": inferred}
        self._log_event("parameter_inference", state, out, "参数推断")
        return out

    async def _initial_retrieval(self, state: Dict[str, Any]) -> Dict[str, Any]:
        breadth = state["params"]["strategy"]["breadth"]
        results = await self.research_agent._initial_bfs_collect(state["parsed"], breadth) if hasattr(self.research_agent, "_initial_bfs_collect") else await self.research_agent._execute_subtasks([{"query": state["parsed"].get("normalized_query", state["query"]) , "type": "bfs"}], parallel=True)
        out = {**state, "initial_results": results}
        self._log_event("initial_retrieval", state, out, "初始信息收集")
        return out

    async def _gap_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        gaps = self.research_agent._analyze_gaps(state["initial_results"]) if hasattr(self.research_agent, "_analyze_gaps") else {"topics": [], "missing": [], "conflicts": []}
        out = {**state, "gaps": gaps}
        self._log_event("gap_analysis", state, out, "缺口分析")
        return out

    async def _followup_retrieval(self, state: Dict[str, Any]) -> Dict[str, Any]:
        depth = state["params"]["strategy"]["depth"]
        iterations = state["params"]["strategy"]["iterations"]
        results: List[Any] = []
        for _ in range(iterations):
            batch = await self.research_agent._targeted_followups(state["parsed"], state["gaps"], depth) if hasattr(self.research_agent, "_targeted_followups") else await self.research_agent._execute_subtasks([{ "query": state["parsed"].get("normalized_query", state["query"]), "type": "followup" }], parallel=True)
            results.append(batch)
        out = {**state, "followup_results": results}
        self._log_event("followup_retrieval", state, out, "跟进检索")
        return out

    async def _synthesis_analysis(self, state: Dict[str, Any]) -> Dict[str, Any]:
        gathered = {"information": []}
        if isinstance(state.get("initial_results"), dict):
            gathered["information"].extend(state["initial_results"].get("information", []))
        for fr in state.get("followup_results", []):
            if isinstance(fr, dict):
                gathered["information"].extend(fr.get("information", []))
        q = state.get("parsed", {}).get("normalized_query", state.get("query", ""))
        synthesis = await self.research_agent._analyze_information(gathered, q) if hasattr(self.research_agent, "_analyze_information") else {"key_findings": [], "themes": [], "contradictions": [], "gaps": state.get("gaps", {})}
        out = {**state, "synthesis": synthesis}
        self._log_event("synthesis_analysis", state, out, "聚合分析")
        return out

    async def _fact_checking(self, state: Dict[str, Any]) -> Dict[str, Any]:
        claims = []
        module = state.get("module") or {}
        if module.get("findings"):
            for f in module.get("findings", [])[:5]:
                claims.append(str(f.get("text", "")))
        else:
            for k in state.get("synthesis", {}).get("key_findings", [])[:5]:
                claims.append(str(k))
        sources = state.get("sources", [])
        fc = await self.research_agent._fact_check(claims, sources) if hasattr(self.research_agent, "_fact_check") else {"status": "pending", "score": 0.0, "details": []}
        out = {**state, "fact_check": fc}
        self._log_event("fact_checking", state, out, "事实核查")
        return out

    async def _credibility_evaluation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        sources = self.research_agent._collect_sources(state.get("initial_results"), state.get("followup_results")) if hasattr(self.research_agent, "_collect_sources") else []
        credibility = self.research_agent._assess_source_credibility(sources) if hasattr(self.research_agent, "_assess_source_credibility") else {"credible": 0, "questionable": 0, "total": 0}
        confidence = self.research_agent._calculate_confidence(state.get("synthesis", {}), sources) if hasattr(self.research_agent, "_calculate_confidence") else 0.5
        out = {**state, "sources": sources, "credibility": credibility, "confidence": confidence}
        self._log_event("credibility_evaluation", state, out, "可信度评估")
        return out

    async def _report_generation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # assemble retrieval path
        path: List[Dict[str, Any]] = []
        tr = []
        if isinstance(state.get("initial_results"), dict):
            tr.append({"phase": "initial", "trace": state["initial_results"].get("trace", [])})
        for fr in state.get("followup_results", []):
            if isinstance(fr, dict):
                tr.append({"phase": "followup", "trace": fr.get("trace", [])})
        collected_infos = []
        if isinstance(state.get("initial_results"), dict):
            collected_infos.extend(state["initial_results"].get("information", []))
        for fr in state.get("followup_results", []):
            if isinstance(fr, dict):
                collected_infos.extend(fr.get("information", []))
        if hasattr(self.research_agent, "_compose_sections_v2"):
            module = await self.research_agent._compose_sections_v2({"information": collected_infos}, state.get("outline", []), state.get("query", ""))
        elif hasattr(self.research_agent, "_compose_sections"):
            module = await self.research_agent._compose_sections({"information": collected_infos}, state.get("outline", []), state.get("query", ""))
        else:
            module = {}
        # 生成文章正文
        article_text = ""
        if hasattr(self.research_agent, "_compose_longform_article"):
            try:
                article_text = await self.research_agent._compose_longform_article(module, state.get("query", ""), state.get("outline", []))
            except Exception:
                article_text = ""
        polished_text = article_text
        if hasattr(self.research_agent, "_polish_article"):
            try:
                polished_text = await self.research_agent._polish_article(article_text, module, state.get("query", ""))
            except Exception:
                pass
        if isinstance(module, dict):
            module["article"] = article_text
            module["final_article"] = polished_text
        result = {
            "query": state["query"],
            "parsed": state.get("parsed", {}),
            "tasks": state.get("tasks", []),
            "analysis": {**state.get("synthesis", {}), "key_findings": [f.get("text") for f in module.get("findings", [])] or state.get("synthesis", {}).get("key_findings", [])},
            "sources": state.get("sources", []),
            "credibility": state.get("credibility", {}),
            "confidence_level": state.get("confidence", 0),
            "params": state.get("params", {}),
            "events": self._events,
            "outline": state.get("outline", []),
            "retrieval_path": tr,
            "module": module,
            "fact_check": state.get("fact_check", {}),
        }
        fmt = getattr(self.config, "DEFAULT_REPORT_FORMAT", "markdown")
        include_sources = True
        report = await self.report_generator.generate_report(result, fmt, include_sources)
        out = {**state, "report": report, "research_result": result}
        self._log_event("report_generation", state, out, "报告生成")
        return out

    async def _evaluation_monitoring(self, state: Dict[str, Any]) -> Dict[str, Any]:
        iterations = state["params"]["strategy"]["iterations"]
        diversity = len(set([s.get("domain", "") for s in state.get("sources", [])]))
        closure = 0.9 if state.get("synthesis", {}).get("contradictions") == [] else 0.7
        warning = False
        if iterations < 3:
            warning = True
        if diversity < 5:
            warning = True
        if closure < 0.9:
            warning = True
        metrics = {"coverage_iterations": iterations, "source_diversity": diversity, "closure": closure, "quality_warning": warning}
        out = {**state, "evaluation_metrics": metrics}
        self._log_event("evaluation_monitoring", state, out, "评估监控", metrics)
        if warning:
            more = await self._followup_retrieval(out)
            more = await self._synthesis_analysis(more)
            sources = self.research_agent._collect_sources(more.get("initial_results", {}), more.get("followup_results", [])) if hasattr(self.research_agent, "_collect_sources") else more.get("sources", [])
            more["sources"] = sources
            diversity2 = len(set([s.get("domain", "") for s in sources]))
            metrics["source_diversity"] = max(diversity, diversity2)
            out = more
            self._log_event("evaluation_monitoring.retry", state, out, "评估监控-追加检索", metrics)
        if self.learning_manager:
            try:
                self.learning_manager.update_after_run(metrics)
            except Exception:
                pass
        return out

    async def _auto_recovery(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = {**state}
        self._log_event("auto_recovery", state, out, "异常恢复")
        return out

    async def run(self, query: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        deadline = 300
        start = time.time()
        state: Dict[str, Any] = {"query": query, "task_id": task_id}
        if LANGGRAPH_AVAILABLE:
            graph = StateGraph(dict)
            graph.add_node("input_parsing", self._input_parsing)
            graph.add_node("task_decomposition", self._task_decomposition)
            graph.add_node("outline_generation", self._outline_generation)
            graph.add_node("parameter_inference", self._parameter_inference)
            graph.add_node("initial_retrieval", self._initial_retrieval)
            graph.add_node("gap_analysis", self._gap_analysis)
            graph.add_node("followup_retrieval", self._followup_retrieval)
            graph.add_node("synthesis_analysis", self._synthesis_analysis)
            graph.add_node("credibility_evaluation", self._credibility_evaluation)
            graph.add_node("fact_checking", self._fact_checking)
            graph.add_node("report_generation", self._report_generation)
            graph.add_node("evaluation_monitoring", self._evaluation_monitoring)
            graph.add_node("auto_recovery", self._auto_recovery)
            graph.set_entry_point("input_parsing")
            graph.add_edge("input_parsing", "task_decomposition")
            graph.add_edge("task_decomposition", "outline_generation")
            graph.add_edge("outline_generation", "parameter_inference")
            graph.add_edge("parameter_inference", "initial_retrieval")
            graph.add_edge("initial_retrieval", "gap_analysis")
            graph.add_edge("gap_analysis", "followup_retrieval")
            graph.add_edge("followup_retrieval", "synthesis_analysis")
            graph.add_edge("synthesis_analysis", "fact_checking")
            graph.add_edge("fact_checking", "credibility_evaluation")
            graph.add_edge("credibility_evaluation", "report_generation")
            graph.add_edge("report_generation", "evaluation_monitoring")
            graph.add_edge("evaluation_monitoring", "auto_recovery")
            graph.add_edge("auto_recovery", END)
            app = graph.compile()
            state = await app.ainvoke(state)
            return state
        else:
            state = await self._input_parsing(state)
            state = await self._task_decomposition(state)
            state = await self._parameter_inference(state)
            state = await self._initial_retrieval(state)
            state = await self._gap_analysis(state)
            state = await self._followup_retrieval(state)
            state = await self._synthesis_analysis(state)
            state = await self._credibility_evaluation(state)
            state = await self._report_generation(state)
            state = await self._evaluation_monitoring(state)
            state = await self._auto_recovery(state)
            if time.time() - start > deadline:
                self._log_event("deadline", {"elapsed": time.time() - start}, state, "超时")
            return state
