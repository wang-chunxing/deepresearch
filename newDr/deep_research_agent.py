import os
import operator
from typing import Annotated, List, TypedDict

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 引入 LangChain 和 LangGraph 相关组件
# [修改] 使用 langchain_openai 来对接支持 OpenAI 协议的豆包模型
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# ==============================================================================
# 1. 配置 API Key
# ==============================================================================
# 优先检查 ARK_API_KEY (火山引擎标准)，其次检查 DOUBAO_API_KEY
api_key = os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")

if not api_key:
    print("⚠️ 警告: 未检测到 ARK_API_KEY 或 DOUBAO_API_KEY。将使用本地模拟 LLM。")

if not tavily_key:
    print("⚠️ 警告: 未检测到 TAVILY_API_KEY。将使用本地模拟搜索结果。")


# ==============================================================================
# 2. 定义状态 (State)
# ==============================================================================
class ResearchState(TypedDict):
    """
    定义 Agent 的运行状态。
    LangGraph 会在节点之间传递这个状态字典。
    """
    topic: str  # 用户输入的初始研究主题
    sub_queries: List[str]  # 规划阶段生成的具体搜索查询列表
    research_data: List[str]  # 收集到的所有研究资料摘要
    final_report: str  # 最终生成的报告内容


# ==============================================================================
# 3. 初始化模型和工具
# ==============================================================================

class _LLMResponse:
    def __init__(self, content: str):
        self.content = content

class _DummyLLM:
    def invoke(self, messages):
        sys = messages[0].content if messages else ""
        human = messages[-1].content if messages else ""
        if "拆解为 3 个初始搜索查询" in sys or "查询列表" in sys:
            topic = human.split("主题:")[-1].strip() if "主题:" in human else "主题"
            return _LLMResponse(f"{topic} 定义\n{topic} 现状\n{topic} 趋势")
        if "提取关键事实" in sys:
            return _LLMResponse("要点1【来源 1】\n要点2【来源 2】")
        if "专业的行业分析师" in sys:
            return _LLMResponse("# 深度研究报告\n\n- 综述\n- 发现\n- 结论")
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


# ==============================================================================
# 4. 定义节点逻辑 (Nodes)
# ==============================================================================

def plan_research(state: ResearchState):
    """
    【节点 1：规划】
    分析用户的主题，生成 3-5 个具体的搜索引擎查询语句。
    """
    print(f"\n--- [1/3] 正在规划研究路径: {state['topic']} ---")

    topic = state["topic"]

    # 定义规划师的 Prompt
    system_prompt = (
        "你是一个高级研究规划师。"
        "你的任务是将用户提供的宽泛研究主题拆解为 3 个具体的、利于搜索引擎理解的子查询语句。"
        "请直接返回查询列表，每行一个，不要包含序号或其他废话。"
        "查询语句应该多样化，涵盖主题的不同方面（例如：背景、现状、争议、未来趋势）。"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"研究主题: {topic}")
    ]

    # 调用 LLM
    response = llm.invoke(messages)

    # 解析结果：按行分割并清理空白
    queries = [line.strip() for line in response.content.split('\n') if line.strip()]

    # 限制查询数量，避免过多消耗 token
    queries = queries[:3]

    print(f"生成的搜索查询: {queries}")

    # 更新状态：存入 sub_queries
    return {"sub_queries": queries}


def execute_search(state: ResearchState):
    """
    【节点 2：执行研究】
    遍历所有子查询，调用搜索工具，并让 LLM 总结每个查询的结果。
    """
    print("\n--- [2/3] 正在执行深度搜索 ---")

    queries = state["sub_queries"]
    all_findings = []

    for query in queries:
        print(f"  -> 正在搜索: {query}")

        try:
            # 1. 执行搜索
            search_results = search_tool.invoke(query)

            # 将搜索结果转换为字符串上下文
            context = "\n".join([f"来源: {res['url']}\n内容: {res['content']}" for res in search_results])

            # 2. 使用 LLM 提取关键信息 (Summarize)
            # 这一步很重要，避免将过多的原始 HTML 噪音传递给下一步
            summary_prompt = (
                "你是一个敏锐的研究员。"
                "请根据以下搜索结果，提取与查询最相关的事实、数据和观点。"
                "请用简洁的中文总结。如果内容无关，请忽略。"
            )

            messages = [
                SystemMessage(content=summary_prompt),
                HumanMessage(content=f"查询: {query}\n\n搜索结果:\n{context}")
            ]

            summary = llm.invoke(messages).content
            all_findings.append(f"### 关于 '{query}' 的研究发现:\n{summary}")

        except Exception as e:
            print(f"  ❌ 搜索 '{query}' 时出错: {e}")
            continue

    # 更新状态：存入 research_data
    return {"research_data": all_findings}


def write_report(state: ResearchState):
    """
    【节点 3：撰写报告】
    汇总所有研究发现，写出一份 Markdown 格式的深度报告。
    """
    print("\n--- [3/3] 正在撰写最终报告 ---")

    topic = state["topic"]
    research_data = state["research_data"]

    # 将所有发现合并为一个大的上下文块
    context_block = "\n\n".join(research_data)

    # 定义撰写者的 Prompt
    system_prompt = (
        "你是一个专业的行业分析师。"
        "你的任务是根据提供的研究资料，为用户撰写一份深度研究报告。"
        "要求："
        "1. 报告标题应包含主题。"
        "2. 结构清晰（引言、核心发现、详细分析、结论）。"
        "3. 引用具体的数据或观点来支持你的论述。"
        "4. 使用 Markdown 格式。"
        "5. 语言风格专业、客观。"
        "6. 如果资料中没有相关信息，请诚实地说明，不要编造。"
    )

    user_prompt = f"研究主题: {topic}\n\n收集到的资料:\n{context_block}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)

    # 更新状态：存入 final_report
    return {"final_report": response.content}


# ==============================================================================
# 5. 构建图 (Graph Construction)
# ==============================================================================

# 初始化工作流图
workflow = StateGraph(ResearchState)

# 添加节点
workflow.add_node("planner", plan_research)
workflow.add_node("researcher", execute_search)
workflow.add_node("writer", write_report)

# 定义边 (Edges) - 这里是一个线性的工作流
# Start -> Planner -> Researcher -> Writer -> End
workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

# 编译图
app = workflow.compile()

# ==============================================================================
# 6. 主程序入口
# ==============================================================================

if __name__ == "__main__":
    # 示例运行
    print("=== Deep Research Agent 启动 (Powered by Doubao) ===")

    user_topic = input("请输入你想研究的主题 (例如: '2024年人工智能在医疗领域的发展趋势'): ")

    if not user_topic:
        user_topic = "2024年人工智能在医疗领域的发展趋势"

    # 初始化输入状态
    initial_state = {"topic": user_topic}

    # 执行图
    # config={"recursion_limit": 10} 防止无限循环，虽然此线性图不需要
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print("FINAL REPORT / 最终报告")
    print("=" * 50 + "\n")

    print(final_state["final_report"])

    # 可选：将报告保存到文件
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(final_state["final_report"])
    print("\n[系统] 报告已保存至 research_report.md")
