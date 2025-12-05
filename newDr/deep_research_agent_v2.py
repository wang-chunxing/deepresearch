import os
import operator
from typing import Annotated, List, TypedDict, Union

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 引入 LangChain 和 LangGraph 组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# ==============================================================================
# 1. 配置 API Key (保持与 V1 一致)
# ==============================================================================
api_key = os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")

if not api_key:
    print("⚠️  警告: 未检测到 ARK/DOUBAO API Key，将使用本地模拟 LLM。")

if not tavily_key:
    print("⚠️  警告: 未检测到 TAVILY_API_KEY，将使用本地模拟搜索结果。")


# ==============================================================================
# 2. 定义状态 (State) - V2 升级版
# ==============================================================================
class ResearchState(TypedDict):
    topic: str  # 原始研究主题
    current_queries: List[str]  # 当前这一轮需要执行的搜索查询
    all_findings: List[str]  # 累积收集到的所有信息 (V2 支持多轮累积)
    loop_count: int  # 当前迭代次数 (防止死循环)
    missing_info: str  # 评估阶段发现的缺失信息 (用于指导下一轮)
    final_report: str  # 最终报告


# ==============================================================================
# 3. 初始化模型和工具
# ==============================================================================
# 豆包模型
class _LLMResponse:
    def __init__(self, content: str):
        self.content = content

class _DummyLLM:
    def invoke(self, messages):
        sys = messages[0].content if messages else ""
        human = messages[-1].content if messages else ""
        if "拆解为 3 个初始搜索查询" in sys or "查询列表" in sys:
            topic = human.split("主题:")[-1].strip() if "主题:" in human else "主题"
            return _LLMResponse(f"{topic} 定义\n{topic} 现状\n{topic} 争议")
        if "提取关键事实" in sys:
            return _LLMResponse("要点1【来源 1】\n要点2【来源 2】\n引用: {'[1]': 'https://example.com/1', '[2]': 'https://example.com/2'}")
        if "苛刻的研究导师" in sys:
            return _LLMResponse("SUFFICIENT")
        if "专业分析师" in sys:
            return _LLMResponse("# 深度报告\n\n- 核心发现\n- 分析\n- 结论")
        return _LLMResponse("示例输出")

class _DummySearch:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
    def invoke(self, query: str):
        return [{"url": f"https://example.com/{i}", "content": f"与{query}相关的示例内容 {i}"} for i in range(1, self.max_results + 1)]

llm = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    temperature=0.1,
) if api_key else _DummyLLM()

search_tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_key) if tavily_key else _DummySearch(max_results=3)

# 最大迭代次数 (防止一直搜个没完)
MAX_LOOPS = 3


# ==============================================================================
# 4. 定义节点逻辑 (Nodes)
# ==============================================================================

def plan_research(state: ResearchState):
    """
    【节点 1：初始规划】
    与 V1 类似，生成初始的一组查询。
    """
    print(f"\n🚀 [启动] 开始研究主题: {state['topic']}")

    system_prompt = (
        "你是一个研究规划专家。请将用户的主题拆解为 3 个初始搜索查询。"
        "查询应涵盖基础定义、现状和主要争议点。"
        "只返回查询列表，每行一个。"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"主题: {state['topic']}")
    ])

    queries = [line.strip() for line in response.content.split('\n') if line.strip()][:3]
    print(f"📋 [规划] 初始查询: {queries}")

    # 初始化状态
    return {
        "current_queries": queries,
        "all_findings": [],
        "loop_count": 0,
        "missing_info": ""
    }


def execute_search(state: ResearchState):
    """
    【节点 2：执行搜索】
    执行 current_queries 中的查询，并将结果追加到 all_findings 中。
    """
    loop_idx = state["loop_count"] + 1
    print(f"\n🔍 [第 {loop_idx} 轮搜索] 正在执行 {len(state['current_queries'])} 个查询...")

    new_findings = []

    for query in state["current_queries"]:
        try:
            # 搜索
            search_results = search_tool.invoke(query)
            # 确保search_results是列表并且正确处理其内容
            if isinstance(search_results, list):
                # 安全地处理搜索结果，防止索引错误
                processed_results = []
                for res in search_results:
                    if isinstance(res, dict) and 'content' in res and 'url' in res:
                        processed_results.append(f"- {res['content']} (来源: {res['url']})")
                context = "\n".join(processed_results)
            else:
                # 处理非预期的返回格式
                context = str(search_results)

            # 总结 (Information Extraction)
            summary_prompt = (
                f"针对查询 '{query}'，从以下搜索结果中提取关键事实、数据和观点。"
                "忽略无关信息。用简洁的中文总结。"
            )
            summary = llm.invoke([
                SystemMessage(content=summary_prompt),
                HumanMessage(content=context)
            ]).content

            new_findings.append(f"【第 {loop_idx} 轮 - {query}】:\n{summary}")

        except Exception as e:
            print(f"  ❌ 查询 '{query}' 失败: {e}")

    # 将新发现追加到现有的发现列表中 (使用 operator.add 逻辑或直接列表相加)
    # 在 LangGraph 中，如果我们返回 key 的值，默认是覆盖。
    # 这里我们手动合并列表返回。
    total_findings = state["all_findings"] + new_findings

    return {"all_findings": total_findings, "loop_count": loop_idx}


def evaluate_findings(state: ResearchState):
    """
    【节点 3 (V2新增)：评估与反思】
    查看当前收集到的所有信息，判断是否足够写报告。
    如果不够，生成新的查询来填补空白。
    """
    print("\n🤔 [评估] 正在检查资料完整性...")

    topic = state["topic"]
    findings_text = "\n\n".join(state["all_findings"])
    loop_count = state["loop_count"]

    # 如果达到最大次数，强制结束
    if loop_count >= MAX_LOOPS:
        print("🛑 [评估] 已达最大迭代次数，停止搜索。")
        return {"missing_info": "sufficient"}  # 标记为足够，迫使进入写作

    # 让 LLM 评估
    system_prompt = (
        "你是一个苛刻的研究导师。"
        "请阅读目前收集到的笔记，判断是否足以撰写关于该主题的深度报告。"
        "如果资料充足，请只回复 'SUFFICIENT'。"
        "如果资料缺失（例如缺少具体数据、反面观点、最新进展），请回复 'MISSING: <缺失内容的描述>'。"
        "不要客气，如果信息太浅显，必须要求继续深挖。"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"研究主题: {topic}\n\n目前笔记:\n{findings_text}")
    ]).content

    if "SUFFICIENT" in response.upper():
        print("✅ [评估] 资料已充足！")
        return {"missing_info": "sufficient"}
    else:
        print(f"⚠️ [评估] 发现缺口: {response}")
        return {"missing_info": response}


def generate_new_queries(state: ResearchState):
    """
    【节点 4 (V2新增)：生成补充查询】
    如果 evaluate 认为信息缺失，这里负责生成针对性的新查询。
    """
    missing_info = state["missing_info"]
    print("\n🔄 [迭代] 正在生成补充查询以填补缺口...")

    system_prompt = (
        "根据缺失的信息描述，生成 2 个具体的搜索引擎查询语句来填补这些空白。"
        "只返回查询列表，每行一个。"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"缺失信息: {missing_info}")
    ])

    new_queries = [line.strip() for line in response.content.split('\n') if line.strip()][:2]
    print(f"🆕 [补充查询] {new_queries}")

    return {"current_queries": new_queries}


def write_report(state: ResearchState):
    """
    【节点 5：撰写报告】
    """
    print("\n✍️ [写作] 正在整合所有资料撰写报告...")

    context = "\n\n".join(state["all_findings"])
    system_prompt = "你是一个专业分析师。请根据提供的海量研究笔记，写出一份结构严谨、数据详实的深度报告(Markdown格式)。"

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"主题: {state['topic']}\n\n笔记:\n{context}")
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
        return "to_writer"
    else:
        return "to_generator"


# 初始化图
workflow = StateGraph[ResearchState, None, ResearchState, ResearchState](ResearchState)

# 添加节点
workflow.add_node("planner", plan_research)
workflow.add_node("researcher", execute_search)
workflow.add_node("evaluator", evaluate_findings)
workflow.add_node("query_generator", generate_new_queries)
workflow.add_node("writer", write_report)

# 构建流程
# 1. 开始 -> 规划
workflow.set_entry_point("planner")
# 2. 规划 -> 搜索
workflow.add_edge("planner", "researcher")
# 3. 搜索 -> 评估
workflow.add_edge("researcher", "evaluator")

# 4. 评估 -> 条件判断 (继续搜 还是 写报告?)
workflow.add_conditional_edges(
    "evaluator",
    should_continue,
    {
        "to_generator": "query_generator",  # 缺信息 -> 生成新查询
        "to_writer": "writer"  # 够了 -> 写报告
    }
)

# 5. 生成新查询 -> 回到搜索 (闭环)
workflow.add_edge("query_generator", "researcher")

# 6. 写报告 -> 结束
workflow.add_edge("writer", END)

app = workflow.compile()

# ==============================================================================
# 6. 运行入口
# ==============================================================================
if __name__ == "__main__":
    print("=== Deep Research Agent V2 (Self-Correcting) ===")
    topic = input("请输入研究主题: ")
    if not topic: topic = "量子计算机在2024年的最新突破"

    initial_state = {"topic": topic}

    # 运行图
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print(final_state["final_report"])

    # 保存文件
    with open("deep_research_v2.md", "w", encoding="utf-8") as f:
        f.write(final_state["final_report"])
