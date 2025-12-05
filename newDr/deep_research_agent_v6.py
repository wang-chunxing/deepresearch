import os
import operator
import asyncio
import re
from typing import Annotated, List, TypedDict, Union

# 引入 LangChain 和 LangGraph 组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
load_dotenv()

# ==============================================================================
# 1. 配置 API Key
# ==============================================================================
api_key = os.environ.get("ARK_API_KEY") or os.environ.get("DOUBAO_API_KEY")
tavily_key = os.environ.get("TAVILY_API_KEY")

if not api_key:
    print("⚠️  警告: 未检测到豆包 API Key/Endpoint，将使用本地模拟 LLM。")
if not tavily_key:
    print("⚠️  警告: 未检测到 TAVILY_API_KEY，将使用本地模拟搜索结果。")


# ==============================================================================
# 2. 定义状态 (State) - V7 升级版 (支持章节迭代和记忆压缩)
# ==============================================================================
class ResearchState(TypedDict):
    topic: str  # 原始研究主题
    topic_category: str  # 主题类型
    current_queries: List[str]  # 当前这一轮需要执行的搜索查询
    all_findings: List[str]  # 累积收集到的所有信息 (详细摘要或压缩后的长期记忆)

    # V6/V7 写作迭代状态
    report_outline: str  # 报告的完整结构大纲 (Markdown string)
    remaining_chapters: List[str]  # 待写作的章节标题列表
    current_chapter: str  # 正在处理的章节标题
    refined_context: str  # 经过精炼和筛选的**当前章节**写作上下文
    report_sections: List[str]  # 已完成的章节内容（Markdown）

    loop_count: int  # 当前迭代次数 (防止死循环)
    missing_info: str  # 评估阶段发现的缺失信息 (用于指导下一轮)
    final_report: str  # 最终报告


# ==============================================================================
# 3. 初始化模型和工具
# ==============================================================================
# 豆包模型 (所有节点都使用异步调用)
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
        if "上下文压缩专家" in sys and "章节标题" in human:
            return _LLMResponse("与章节直接相关的要点【来源 1】")
        if "专业报告撰稿人" in sys:
            return _LLMResponse("段落内容，包含引用【来源 1】")
        return _LLMResponse("示例输出")

class _DummySearch:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
    async def ainvoke(self, query: str):
        return [{"url": f"https://example.com/{i}", "content": f"与{query}相关的示例内容 {i}"} for i in range(1, self.max_results + 1)]

llm = ChatOpenAI(
    model="doubao-seed-1-6-251015",
    api_key=api_key,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    temperature=0.1,
) if api_key else _DummyLLM()

search_tool = TavilySearchResults(max_results=3) if tavily_key else _DummySearch(max_results=3)

# 最大迭代次数
MAX_LOOPS = 3
# 记忆压缩阈值 (all_findings 超过此数量后触发压缩)
COMPRESSION_THRESHOLD = 6


# ==============================================================================
# 4. 定义节点逻辑 (Nodes) - 全部改为 async
# ==============================================================================

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

    queries_response = await llm.ainvoke([
        SystemMessage(content=planning_prompt),
        HumanMessage(content=f"主题: {state['topic']}")
    ])
    queries = [line.strip() for line in queries_response.content.split('\n') if line.strip()][:3]

    # 步骤 B: 主题分类
    categorization_prompt = (
        "根据用户的主题，将其归类为以下类型之一：[技术综述, 市场分析, 经济趋势, 历史事件, 人物传记, 行业报告, 概念解释]。"
        "请只返回最合适的类别名称，不带任何解释或标点符号。"
    )
    category_response = await llm.ainvoke([
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
        "missing_info": "",
        "report_sections": []
    }


async def execute_search(state: ResearchState):
    """
    【节点 2：执行搜索】 - 并发搜索，确保摘要中嵌入了来源URL。
    """
    loop_idx = state["loop_count"] + 1
    queries = state["current_queries"]
    print(f"\n🔍 [第 {loop_idx} 轮搜索] 正在并发执行 {len(queries)} 个查询...")

    # 简单实现指数退避 (Exponential Backoff) 机制
    async def api_call_with_retry(llm_input, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await llm.ainvoke(llm_input)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt  # 1s, 2s, 4s...
                    print(f"   ⚠️ API 调用失败，尝试 {attempt + 1}/{max_retries}，等待 {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise e
        return None

    async def process_query(query):
        """异步执行单个查询和总结的子任务"""
        try:
            # 1. 异步搜索
            search_results = await search_tool.ainvoke(query)

            # 准备上下文和引用映射
            context_blocks = []
            citation_map = {}
            for i, res in enumerate(search_results):
                # 为每个来源分配一个临时编号用于摘要引用
                citation_map[f"[{i + 1}]"] = res['url']
                context_blocks.append(f"【来源 {i + 1}】: {res['content']}")

            context = "\n".join(context_blocks)

            # 2. 异步总结 (Information Extraction) - 使用重试机制
            summary_prompt = (
                f"针对查询 '{query}'，从以下【来源】中提取关键事实、数据和观点。"
                "在摘要中，务必使用格式【来源 X】引用你使用的任何信息，例如：'人工智能投资在2023年增长了30%【来源 2】'。"
                "忽略无关信息。用简洁的中文总结，并列出完整的引用映射。"
                f"最终返回格式：\n---\n摘要内容\n---\n引用: {citation_map}"
            )

            summary_response = await api_call_with_retry([
                SystemMessage(content=summary_prompt),
                HumanMessage(content=context)
            ])

            # 提取摘要和引用，并将其合并成一个 V4 格式的发现块
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


async def recursive_summarizer(state: ResearchState):
    """
    【节点 2.5 (V7 新增)：递归记忆压缩】
    模拟 MemGPT 机制，周期性地压缩旧的、详细的发现，保持上下文精简。
    """
    findings = state["all_findings"]

    # 检查是否达到压缩阈值
    if len(findings) < COMPRESSION_THRESHOLD:
        print("💡 [记忆] 发现项不足，跳过递归摘要。")
        return {}

    # 1. 识别需要压缩的“旧”记忆 (压缩列表的前半部分)
    split_point = len(findings) // 2
    old_findings_to_compress = findings[:split_point]
    new_findings_to_keep = findings[split_point:]

    print(f"\n🧠 [记忆压缩] 发现 {len(findings)} 项，正在压缩前 {split_point} 项 (旧记忆)...")

    context_to_summarize = "\n\n---\n\n".join(old_findings_to_compress)

    system_prompt = (
        "你是一个递归记忆压缩引擎。你的任务是将下方提供的旧研究发现，压缩成一个**单一、高层次、浓缩的摘要**。"
        "目标是保留核心事实和趋势，但移除不必要的细节，以便为新的研究发现腾出内存空间。"
        "压缩后的摘要必须以 '### 递归长期记忆摘要:' 开头。"
        "请务必在摘要中保留所有原始的【来源 X】引用标记。"
        "只返回摘要内容，不带其他引导词。"
    )

    # 使用重试机制调用 LLM
    async def api_call_with_retry(llm_input, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await llm.ainvoke(llm_input)
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = 2 ** attempt
                    await asyncio.sleep(delay)
                else:
                    raise e
        return None

    response = await api_call_with_retry([
        SystemMessage(content=system_prompt),
        HumanMessage(content=context_to_summarize)
    ])

    compressed_summary = response.content.strip()

    # 2. 更新 all_findings: [压缩后的摘要] + [新的/未压缩的发现]
    new_all_findings = [compressed_summary] + new_findings_to_keep

    print(f"✅ [记忆] 记忆压缩完成。当前发现项数量: {len(new_all_findings)}")

    return {"all_findings": new_all_findings}


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
        "请阅读目前收集到的笔记（包括递归摘要），判断是否足以撰写关于该主题的深度报告。"
        "如果资料充足，请只回复 'SUFFICIENT'。"
        "如果资料缺失（例如缺少具体数据、反面观点、最新进展），请回复 'MISSING: <缺失内容的描述>'。"
        "不要客气，如果信息太浅显，必须要求继续深挖。"
    )

    response = await llm.ainvoke([
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

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"缺失信息: {missing_info}")
    ])

    new_queries = [line.strip() for line in response.content.split('\n') if line.strip()][:2]
    print(f"🆕 [补充查询] {new_queries}")

    return {"current_queries": new_queries}


async def outline_report(state: ResearchState):
    """
    【节点 5：动态生成报告大纲】 - 根据主题类别和初步发现生成大纲。
    """
    print("\n📐 [结构] 正在生成动态报告大纲...")

    topic = state["topic"]
    category = state["topic_category"]
    # 传入压缩后的全部发现
    findings_preview = "\n\n".join(state["all_findings"])

    system_prompt = (
        f"你是一个高级报告结构师。主题类别是 '{category}'。"
        "请根据这个类别和以下初步研究笔记，生成一份最专业、最相关的报告大纲。"
        "大纲应至少包含 4 个主要章节（Markdown 二级标题 ##），并直接返回 Markdown 格式的大纲。"
        "注意：不要在二级标题中包含 '引言' 或 '结论'，留给后面的节点处理。"
    )

    user_prompt = f"研究主题: {topic}\n主题类别: {category}\n\n初步研究笔记:\n{findings_preview}"

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    print(f"✅ [结构] 大纲已生成，基于类别: {category}。")
    return {"report_outline": response.content}


async def parse_outline_and_prepare_chapters(state: ResearchState):
    """
    【节点 6：解析大纲并准备章节】
    将完整的Markdown大纲解析为待写作的章节列表，并设置第一个章节。
    """
    print("\n📚 [准备] 正在解析大纲并准备迭代写作...")

    outline = state["report_outline"]

    # 使用正则表达式匹配所有 Markdown 二级标题
    chapter_titles = re.findall(r'##\s*(.*)', outline)

    # 增加标准的引言和结论作为迭代的首尾章节
    all_chapters = ["引言"] + chapter_titles + ["结论"]

    if not all_chapters:
        return {"remaining_chapters": [], "current_chapter": ""}

    # 弹出第一个作为当前章节
    current_chapter = all_chapters.pop(0)

    print(f"➡️ [当前章节] '{current_chapter}' | 剩余 {len(all_chapters)} 章待写。")

    return {
        "remaining_chapters": all_chapters,
        "current_chapter": current_chapter,
    }


async def chapter_context_retriever(state: ResearchState):
    """
    【节点 7: 章节上下文检索器】
    根据当前章节标题和所有资料，提炼出最关键的上下文。
    """
    print(f"\n✂️ [检索] 正在为章节 '{state['current_chapter']}' 提炼核心上下文...")

    chapter_title = state["current_chapter"]
    all_findings = "\n\n".join(state["all_findings"])

    # 提炼指令更加精确，聚焦于当前章节
    system_prompt = (
        "你是一个上下文压缩专家。你的任务是根据给定的**章节标题**，从下方所有研究发现（包含详细发现和长期记忆摘要）中，"
        "仅挑选出**与该章节主题直接相关**的事实、数据和引用信息。"
        "请将提炼后的信息以精简、结构化的方式返回，**务必保留所有【来源 X】标记**。"
        "目标：将上下文压缩到最精简，只保留撰写该章节所需的核心论据。"
    )

    user_prompt = f"章节标题: {chapter_title}\n\n全部原始研究发现:\n{all_findings}"

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {"refined_context": response.content}


async def chapter_writer(state: ResearchState):
    """
    【节点 8：章节撰写器】
    使用精炼后的上下文，只撰写当前章节的内容。
    """
    chapter_title = state["current_chapter"]
    refined_context = state["refined_context"]

    print(f"\n✍️ [写作] 正在撰写章节: '{chapter_title}'...")

    system_prompt = (
        "你是一个专业报告撰稿人。请根据提供的**精炼上下文**，撰写关于主题的**一个独立章节**。"
        "如果章节是'引言'或'结论'，请相应调整写作风格。"
        "如果章节是主体内容，请将标题作为 Markdown 二级标题（例如：## 章节标题）开头，然后撰写内容。"
        "如果章节是'引言'或'结论'，请只撰写内容，不要添加二级标题。"
        "写作时，必须使用上下文中的【来源 X】标记。"
        "只输出章节内容，不要添加任何其他评论或引导词。"
    )

    user_prompt = (
        f"报告主题: {state['topic']}\n"
        f"当前章节标题: {chapter_title}\n\n"
        f"章节写作上下文:\n{refined_context}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    # 组装完整的章节内容，包括标题
    if chapter_title in ["引言", "结论"]:
        chapter_content = response.content.strip()
    else:
        chapter_content = f"## {chapter_title}\n\n{response.content.strip()}"

    # 将完成的章节内容添加到报告部分列表
    new_sections = state["report_sections"] + [chapter_content]

    # 设置下一个章节
    next_chapter = state["remaining_chapters"].pop(0) if state["remaining_chapters"] else ""

    print(f"✅ [完成] 章节 '{chapter_title}' 写入完毕。")
    return {
        "report_sections": new_sections,
        "current_chapter": next_chapter,
        "remaining_chapters": state["remaining_chapters"],
    }


async def finalize_report(state: ResearchState):
    """
    【节点 9：最终报告整合】
    将所有章节内容和参考资料整合为最终报告。
    """
    print("\n📝 [整合] 正在整合所有章节和参考资料...")

    # 1. 提取所有引用
    full_text = "\n\n".join(state["report_sections"]) + "\n\n" + "\n\n".join(state["all_findings"])

    # 查找所有引用映射 (V4/V5/V6/V7 格式：引用: {..., "['1']": "url"})
    citation_pattern = re.compile(r"引用: (\{.*?\}|\{.*?\})", re.DOTALL)

    unique_references = {}

    for match in citation_pattern.finditer(full_text):
        try:
            # 这是一个简化的JSON解析，实际需要更健壮的逻辑
            citation_str = match.group(1).replace("'", '"')
            citation_map = eval(citation_str)  # 使用 eval 简化，但在生产中应避免
            unique_references.update(citation_map)
        except Exception as e:
            # 忽略解析失败的引用
            continue

    # 2. 格式化参考资料
    references_list = []
    if unique_references:
        references_list.append("## 参考资料 (Citations)")
        # 按照引用 ID 排序 (e.g., [1], [2]...)
        # 注意: 这里的键可能包含 '### 递归长期记忆摘要:'，需要过滤
        valid_refs = [(k, v) for k, v in unique_references.items() if k.startswith('[') and k.endswith(']')]

        # 使用安全的排序，确保键是数字
        def get_sort_key(item):
            try:
                return int(item[0].strip('[]'))
            except ValueError:
                return float('inf')  # 将非数字键放在最后

        sorted_refs = sorted(valid_refs, key=get_sort_key)

        for source_id, url in sorted_refs:
            references_list.append(f"{source_id} {url}")

    # 3. 组合最终报告
    report_title = f"# {state['topic']} 深度研究报告\n\n"
    final_report = report_title + "\n\n".join(state["report_sections"]) + "\n\n" + "\n".join(references_list)

    print("✅ [完成] 最终报告整合完毕。")
    return {"final_report": final_report}


# ==============================================================================
# 5. 构建图逻辑 (Routing Logic)
# ==============================================================================

def should_continue_research(state: ResearchState):
    """
    条件边逻辑：决定是回去接着搜，还是进入写作流程
    """
    missing = state.get("missing_info", "")
    if missing == "sufficient" or state["loop_count"] >= MAX_LOOPS:
        return "to_outline"
    else:
        return "to_generator"


def should_continue_writing(state: ResearchState):
    """
    条件边逻辑：决定是继续写下一章，还是结束写作
    """
    if state["current_chapter"]:
        return "continue_chapter"
    else:
        return "finalize"


# 初始化图
workflow = StateGraph(ResearchState)

# 添加节点
workflow.add_node("planner", plan_research)
workflow.add_node("researcher", execute_search)
workflow.add_node("summarizer", recursive_summarizer)  # [V7 新增] 递归摘要节点
workflow.add_node("evaluator", evaluate_findings)
workflow.add_node("query_generator", generate_new_queries)
workflow.add_node("outline_planner", outline_report)
workflow.add_node("parse_chapters", parse_outline_and_prepare_chapters)
workflow.add_node("chapter_context_retriever", chapter_context_retriever)
workflow.add_node("chapter_writer", chapter_writer)
workflow.add_node("finalizer", finalize_report)

# 构建流程：研究阶段
workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")

# V7 流程修改：搜索 -> 摘要 -> 评估
workflow.add_edge("researcher", "summarizer")  # 搜索结果先进入摘要压缩
workflow.add_edge("summarizer", "evaluator")

# 评估 -> 条件判断
workflow.add_conditional_edges(
    "evaluator",
    should_continue_research,
    {
        "to_generator": "query_generator",
        "to_outline": "outline_planner"
    }
)

# 迭代循环
workflow.add_edge("query_generator", "researcher")

# 结构化写作阶段
workflow.add_edge("outline_planner", "parse_chapters")
workflow.add_edge("parse_chapters", "chapter_context_retriever")
workflow.add_edge("chapter_context_retriever", "chapter_writer")

# 章节写作 -> 条件判断
workflow.add_conditional_edges(
    "chapter_writer",
    should_continue_writing,
    {
        "continue_chapter": "chapter_context_retriever",  # 循环到下一章
        "finalize": "finalizer"  # 结束写作
    }
)

workflow.add_edge("finalizer", END)

app = workflow.compile()


# ==============================================================================
# 6. 运行入口
# ==============================================================================

async def run_agent():
    print("=== Deep Research Agent V7 (递归记忆压缩架构) ===")
    topic = input("请输入研究主题: ")
    if not topic: topic = "量子计算机在2024年的最新突破"

    initial_state = {"topic": topic}

    final_state = await app.ainvoke(initial_state)

    print("\n" + "=" * 50)
    print("最终报告:")
    print(final_state["final_report"])

    # 保存文件
    with open("deep_research_v7.md", "w", encoding="utf-8") as f:
        f.write(final_state["final_report"])
    print("\n[系统] 报告已保存至 deep_research_v7.md")


if __name__ == "__main__":
    asyncio.run(run_agent())
