import asyncio
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 延迟导入业务模块，确保 CLI 参数设置的环境变量生效


async def run(query: str, fmt: str = "markdown", outfile: str = None):
    import importlib
    cfg = importlib.import_module("config")
    from src.memory.memory_manager import MemoryManager
    from src.memory.learning_manager import LearningManager
    from src.generation.report_generator import ReportGenerator
    from src.agents.research_agent import ResearchAgent
    from src.workflows.research_graph import ResearchGraphRunner
    mm = MemoryManager()
    lm = LearningManager(cfg.CHROMA_PERSIST_DIR)
    rg = ReportGenerator()
    ra = ResearchAgent(mm, rg, learning_manager=lm)
    runner = ResearchGraphRunner(ra, mm, rg, cfg, learning_manager=lm)
    state = await runner.run(query, task_id=f"cli_{int(datetime.now().timestamp())}")
    report = state.get("report", "")
    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved report to {outfile}")
    else:
        print(report)


def main():
    parser = argparse.ArgumentParser(description="DeepResearch CLI")
    parser.add_argument("query", help="研究主题，如公司或技术点")
    parser.add_argument("--format", dest="fmt", default="markdown", choices=["markdown", "html"], help="报告格式")
    parser.add_argument("--out", dest="outfile", default=None, help="输出文件路径")
    parser.add_argument("--provider", dest="provider", default=None, choices=["doubao", "openai"], help="LLM 提供方覆盖")
    parser.add_argument("--ark_key", dest="ark_key", default=None, help="豆包 ARK_API_KEY")
    parser.add_argument("--doubao_endpoint", dest="doubao_endpoint", default=None, help="豆包 API 端点")
    parser.add_argument("--doubao_model", dest="doubao_model", default=None, help="豆包模型名")
    args = parser.parse_args()
    # 覆盖环境变量优先级
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.ark_key:
        os.environ["ARK_API_KEY"] = args.ark_key
    if args.doubao_endpoint:
        os.environ["DOUBAO_API_ENDPOINT"] = args.doubao_endpoint
    if args.doubao_model:
        os.environ["DOUBAO_MODEL"] = args.doubao_model
    asyncio.run(run(args.query, fmt=args.fmt, outfile=args.outfile))


if __name__ == "__main__":
    main()
