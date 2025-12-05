import os
import operator
import asyncio  # [V3 优化] 引入 asyncio 实现并发
from typing import Annotated, List, TypedDict, Union

# 引入 LangChain 和 LangGraph 组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

# ==============================================================================
# 1. 配置 API Key (保持与 V1 一致)
# ==============================================================================
api_key = os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
endpoint_id = os.environ.get("DOUBAO_ENDPOINT_ID")
tavily_key = os.environ.get("TAVILY_API_KEY")
offline = os.environ.get("DR_OFFLINE") == "1"
timeout_env = os.environ.get("DR_TIMEOUT")
DEFAULT_TIMEOUT_S = int(timeout_env) if timeout_env and timeout_env.isdigit() else 30

if not api_key and not endpoint_id:
    print("⚠️  警告: 未检测到豆包 API Key/Endpoint，将使用本地模拟 LLM。")
if not tavily_key:
    print("⚠️  警告: 未检测到 TAVILY_API_KEY，将使用本地模拟搜索结果。")


# ==============================================================================
# 2. 定义状态 (State) - V4 升级版
# ==============================================================================
class ResearchState(TypedDict):
    topic: str  # 原始研究主题
    topic_category: str  # [V4 新增] 主题类型 (e.g., 经济分析, 技术综述)
    current_queries: List[str]  # 当前这一轮需要执行的搜索查询
    all_findings: List[str]  # 累积收集到的所有信息 (包含摘要和引用URL)
    loop_count: int  # 当前迭代次数 (防止死循环)
    missing_info: str  # 评估阶段发现的缺失信息 (用于指导下一轮)
    report_outline: str  # 报告的结构大纲 (现在是动态生成)
    final_report: str  # 最终报告


# ==============================================================================
# 3. 初始化模型和工具
# ==============================================================================
# 豆包模型 (所有节点都使用异步调用，因此所有节点函数都改为 async)
class _LLMResponse:
    def __init__(self, content: str):
        self.content = content

class _DummyLLM:
    async def ainvoke(self, messages):
        sys = messages[0].content if messages else ""
        human = messages[-1].content if messages else ""
        if "拆解为 3 个初始搜索查询" in sys or "查询列表" in sys:
            topic = human.split("主题:")[-1].strip() if "主题:" in human else "主题"
            return _LLMResponse(f"{topic} 定义\n{topic} 现状\n{topic} 趋势")
        if "归类为以下类型之一" in sys:
            return _LLMResponse("技术综述")
        if "提取关键事实" in sys:
            return _LLMResponse("要点1【来源 1】\n要点2【来源 2】\n---\n引用: {'[1]': 'https://example.com/1', '[2]': 'https://example.com/2'}")
        if "苛刻的研究导师" in sys:
            return _LLMResponse("SUFFICIENT")
        if "高级报告结构师" in sys:
            return _LLMResponse("## 背景\n## 现状\n## 竞争格局\n## 趋势")
        if "专业分析师" in sys:
            return _LLMResponse("# 报告\n\n## 背景\n内容\n\n## 现状\n内容\n\n## 竞争格局\n内容\n\n## 趋势\n内容")
        return _LLMResponse("示例输出")

class _DummySearch:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
    async def ainvoke(self, query: str):
        return [{"url": f"https://example.com/{i}", "content": f"与{query}相关的示例内容 {i}"} for i in range(1, self.max_results + 1)]

if api_key and not offline:
    model_id = endpoint_id or os.environ.get("DOUBAO_MODEL") or "doubao-seed-1-6-251015"
    llm = ChatOpenAI(
        model=model_id,
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        temperature=0.1,
    )
else:
    llm = _DummyLLM()

search_tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_key) if tavily_key else _DummySearch(max_results=3)

# 回退实例
_FALLBACK_LLM = _DummyLLM()
_FALLBACK_SEARCH = _DummySearch(max_results=3)

# 最大迭代次数 (防止一直搜个没完)
MAX_LOOPS = 3


# ==============================================================================
# 4. 定义节点逻辑 (Nodes) - 全部改为 async
# ==============================================================================

# 统一异步调用封装，兼容 invoke/ainvoke 两种实现
async def _llm_call(messages, timeout_s: int = DEFAULT_TIMEOUT_S):
    fn = getattr(llm, "ainvoke", None)
    if callable(fn):
        try:
            return await asyncio.wait_for(fn(messages), timeout_s)
        except Exception as e:
            print(f"[警告] LLM 调用超时/失败，使用本地回退。错误: {e}")
            return await _FALLBACK_LLM.ainvoke(messages)
    invoke_fn = getattr(llm, "invoke", None)
    if callable(invoke_fn):
        try:
            return await asyncio.wait_for(asyncio.to_thread(invoke_fn, messages), timeout_s)
        except Exception as e:
            print(f"[警告] LLM 调用超时/失败，使用本地回退。错误: {e}")
            return await _FALLBACK_LLM.ainvoke(messages)
    raise RuntimeError("LLM 不支持 invoke/ainvoke")

async def _search_call(query: str, timeout_s: int = DEFAULT_TIMEOUT_S):
    fn = getattr(search_tool, "ainvoke", None)
    if callable(fn):
        try: 
            return await asyncio.wait_for(fn(query), timeout_s)
        except Exception as e:
            print(f"[警告] 搜索调用超时/失败，使用本地回退。错误: {e}")
            return await _FALLBACK_SEARCH.ainvoke(query)
    invoke_fn = getattr(search_tool, "invoke", None)
    if callable(invoke_fn):
        try:
            return await asyncio.wait_for(asyncio.to_thread(invoke_fn, query), timeout_s)
        except Exception as e:
            print(f"[警告] 搜索调用超时/失败，使用本地回退。错误: {e}")
            return await _FALLBACK_SEARCH.ainvoke(query)
    raise RuntimeError("搜索工具不支持 invoke/ainvoke")

async def plan_research(state: ResearchState):
    """
    【节点 1：初始规划与分类】
    生成初始查询并对主题进行分类，用于指导后续的大纲生成。
    """
    print(f"\n🚀 [启动] 开始研究主题: {state['topic']}")

    # 步骤 A: 生成初始查询
    planning_prompt = (
        "你是一个研究规划专家。请将用户的主题拆解为 3 个初始搜索查询。"
        "查询应涵盖基础定义、现状和主要争议点。"
        "只返回查询列表，每行一个。"
    )

    queries_response = await _llm_call([
        SystemMessage(content=planning_prompt),
        HumanMessage(content=f"主题: {state['topic']}")
    ])
    queries = [line.strip() for line in queries_response.content.split('\n') if line.strip()][:3]

    # 步骤 B: 主题分类 (V4 新增)
    categorization_prompt = (
        "根据用户的主题，将其归类为以下类型之一：[技术综述, 市场分析, 经济趋势, 历史事件, 人物传记, 行业报告, 概念解释]。"
        "请只返回最合适的类别名称，不带任何解释或标点符号。"
    )
    category_response = await _llm_call([
        SystemMessage(content=categorization_prompt),
        HumanMessage(content=f"主题: {state['topic']}")
    ])
    topic_category = category_response.content.strip()

    print(f"📋 [规划] 主题类型: {topic_category} | 初始查询: {queries}")

    # 初始化状态
    return {
        "current_queries": queries,
        "topic_category": topic_category,
        "all_findings": [],
        "loop_count": 0,
        "missing_info": ""
    }


async def execute_search(state: ResearchState):
    """
    【节点 2：执行搜索】 - V4: 确保摘要中嵌入了来源URL。
    """
    loop_idx = state["loop_count"] + 1
    queries = state["current_queries"]
    print(f"\n🔍 [第 {loop_idx} 轮搜索] 正在并发执行 {len(queries)} 个查询...")

    async def process_query(query):
        """异步执行单个查询和总结的子任务"""
        try:
            # 1. 异步搜索
            search_results = await _search_call(query)

            # 准备上下文和引用映射（鲁棒解析）
            context_blocks = []
            citation_map = {}
            items = search_results if isinstance(search_results, list) else [search_results]
            for i, res in enumerate(items):
                url = None
                content = None
                if isinstance(res, dict):
                    url = res.get('url') or res.get('source') or res.get('link')
                    content = res.get('content') or res.get('text') or res.get('summary')
                else:
                    url = getattr(res, 'url', None) or getattr(res, 'source', None) or getattr(res, 'link', None)
                    content = getattr(res, 'content', None) or getattr(res, 'page_content', None) or getattr(res, 'text', None)

                if not content:
                    content = str(res)

                if url:
                    citation_map[f"[{i + 1}]"] = url
                context_blocks.append(f"【来源 {i + 1}】: {content}")

            context = "\n".join(context_blocks)

            # 2. 异步总结 (Information Extraction)
            summary_prompt = (
                f"针对查询 '{query}'，从以下【来源】中提取关键事实、数据和观点。"
                "在摘要中，务必使用格式【来源 X】引用你使用的任何信息，例如：'人工智能投资在2023年增长了30%【来源 2】'。"
                "忽略无关信息。用简洁的中文总结，并列出完整的引用映射。"
                f"最终返回格式：\n---\n摘要内容\n---\n引用: {citation_map}"
            )

            summary_response = await _llm_call([
                SystemMessage(content=summary_prompt),
                HumanMessage(content=context)
            ])

            # 提取摘要和引用，并将其合并成一个 V4 格式的发现块
            # 示例 V4 发现块： ### 查询 X 发现\n 摘要内容 \n\n --- 引用映射: {...}
            return f"### 关于 '{query}' 的发现 (第 {loop_idx} 轮):\n{summary_response.content}"

        except Exception as e:
            print(f"  ❌ 查询 '{query}' 失败: {e}")
            return None

    # 并行执行所有查询任务
    tasks = [process_query(query) for query in queries]
    results = await asyncio.gather(*tasks)

    new_findings = [r for r in results if r]
    total_findings = state["all_findings"] + new_findings

    return {"all_findings": total_findings, "loop_count": loop_idx}


async def evaluate_findings(state: ResearchState):
    """
    【节点 3：评估与反思】
    查看当前收集到的所有信息，判断是否足够写报告。
    """
    print("\n🤔 [评估] 正在检查资料完整性...")

    topic = state["topic"]
    findings_text = "\n\n".join(state["all_findings"])
    loop_count = state["loop_count"]

    if loop_count >= MAX_LOOPS:
        print("🛑 [评估] 已达最大迭代次数，停止搜索。")
        return {"missing_info": "sufficient"}

        # 让 LLM 评估
    system_prompt = (
        "你是一个苛刻的研究导师。"
        "请阅读目前收集到的笔记，判断是否足以撰写关于该主题的深度报告。"
        "如果资料充足，请只回复 'SUFFICIENT'。"
        "如果资料缺失（例如缺少具体数据、反面观点、最新进展），请回复 'MISSING: <缺失内容的描述>'。"
        "不要客气，如果信息太浅显，必须要求继续深挖。"
    )

    response = await _llm_call([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"研究主题: {topic}\n\n目前笔记:\n{findings_text}")
    ])

    if "SUFFICIENT" in response.content.upper():
        print("✅ [评估] 资料已充足！")
        return {"missing_info": "sufficient"}
    else:
        print(f"⚠️ [评估] 发现缺口: {response.content}")
        return {"missing_info": response.content}


async def generate_new_queries(state: ResearchState):
    """
    【节点 4：生成补充查询】
    如果 evaluate 认为信息缺失，这里负责生成针对性的新查询。
    """
    missing_info = state["missing_info"]
    print("\n🔄 [迭代] 正在生成补充查询以填补缺口...")

    system_prompt = (
        "根据缺失的信息描述，生成 2 个具体的搜索引擎查询语句来填补这些空白。"
        "只返回查询列表，每行一个。"
    )

    response = await _llm_call([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"缺失信息: {missing_info}")
    ])

    new_queries = [line.strip() for line in response.content.split('\n') if line.strip()][:2]
    print(f"🆕 [补充查询] {new_queries}")

    return {"current_queries": new_queries}


async def outline_report(state: ResearchState):
    """
    【节点 5：动态生成报告大纲】 - V4: 根据主题类别和初步发现生成大纲。
    """
    print("\n📐 [结构] 正在生成动态报告大纲...")

    topic = state["topic"]
    category = state["topic_category"]
    findings_preview = "\n\n".join(state["all_findings"])[:2000]  # 传递部分发现作为上下文

    system_prompt = (
        f"你是一个高级报告结构师。主题类别是 '{category}'。"
        "请根据这个类别和以下初步研究笔记，生成一份最专业、最相关的报告大纲。"
        "例如，如果是'市场分析'，大纲应包括'市场规模'、'竞争格局'、'SWOT'；如果是'历史事件'，则应包含'背景'、'过程'、'影响'。"
        "大纲应至少包含 4 个主要章节（Markdown 二级标题 ##），并直接返回 Markdown 格式的大纲。"
    )

    user_prompt = f"研究主题: {topic}\n主题类别: {category}\n\n初步研究笔记预览:\n{findings_preview}"

    response = await _llm_call([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    print(f"✅ [结构] 大纲已生成，基于类别: {category}。")
    return {"report_outline": response.content}


async def write_report(state: ResearchState):
    """
    【节点 6：撰写报告】 - V4: 必须根据嵌入的引用信息，在报告末尾列出参考资料。
    """
    print("\n✍️ [写作] 正在整合所有资料撰写报告...")

    context = "\n\n".join(state["all_findings"])
    outline = state["report_outline"]

    system_prompt = (
        "你是一个专业分析师。请根据提供的研究笔记和以下大纲，写出一份结构严谨、数据详实的深度报告(Markdown格式)。"
        "严格遵循大纲结构。"
        "写作时，必须参考笔记中的【来源 X】标记，并将这些引用信息在报告末尾的'参考资料'部分完整列出，使用标准URL格式。"
    )

    user_prompt = f"主题: {state['topic']}\n\n结构大纲:\n{outline}\n\n研究笔记:\n{context}"

    response = await _llm_call([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {"final_report": response.content}


# ==============================================================================
# 5. 构建图逻辑 (Routing Logic)
# ==============================================================================

def should_continue(state: ResearchState):
    """
    条件边逻辑：决定是回去接着搜，还是去写报告
    """
    missing = state.get("missing_info", "")
    if missing == "sufficient" or state["loop_count"] >= MAX_LOOPS:
        return "to_outline"
    else:
        return "to_generator"


# 初始化图
workflow = StateGraph(ResearchState)

# 添加节点
workflow.add_node("planner", plan_research)
workflow.add_node("researcher", execute_search)
workflow.add_node("evaluator", evaluate_findings)
workflow.add_node("query_generator", generate_new_queries)
workflow.add_node("outline_planner", outline_report)
workflow.add_node("writer", write_report)

# 构建流程
workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "evaluator")

# 评估 -> 条件判断
workflow.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "to_generator": "query_generator",
        "to_outline": "outline_planner"
    }
)

# 迭代循环
workflow.add_edge("query_generator", "researcher")

# 结构化写作
workflow.add_edge("outline_planner", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()


# ==============================================================================
# 6. 运行入口
# ==============================================================================

async def run_agent():
    print("=== Deep Research Agent V4 (动态结构 & 引用溯源) ===")
    topic = input("请输入研究主题: ")
    if not topic: topic = "量子计算机在2024年的最新突破"

    initial_state = {"topic": topic}

    final_state = await app.ainvoke(initial_state)

    print("\n" + "=" * 50)
    print("最终报告:")
    print(final_state["final_report"])

    # 保存文件
    with open("deep_research_v4.md", "w", encoding="utf-8") as f:
        f.write(final_state["final_report"])
    print("\n[系统] 报告已保存至 deep_research_v4.md")


if __name__ == "__main__":
    asyncio.run(run_agent())

async def run_agent_once(topic: str):
    initial_state = {"topic": topic}
    final_state = await app.ainvoke(initial_state)
    return final_state
