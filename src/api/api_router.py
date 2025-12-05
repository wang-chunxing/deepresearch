"""
API路由器用于深度研究报告生成代理系统
基于架构文档中的交互层设计
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import asyncio
from datetime import datetime

from src.agents.research_agent import ResearchAgent
from src.memory.memory_manager import MemoryManager
from src.generation.report_generator import ReportGenerator

logger = logging.getLogger(__name__)
api_router = APIRouter()

# 请求/响应模型
class ResearchRequest(BaseModel):
    query: str = Field(..., description="要研究的查询或主题")


class ResearchResponse(BaseModel):
    task_id: str
    status: str
    query: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    report: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    message: str
    completed_at: Optional[datetime] = None


# 内存任务存储（在生产环境中，使用数据库）
task_storage: Dict[str, Dict[str, Any]] = {}


@api_router.post("/research", response_model=ResearchResponse, tags=["research"])
async def start_research(request: ResearchRequest):
    """
    根据提供的查询启动新的研究任务。
    这实现了我们架构中的交互层。
    """
    from main import research_agent  # 在此处导入以避免循环导入
    
    task_id = f"task_{int(datetime.now().timestamp())}_{hash(request.query) % 10000}"
    
    # 存储初始任务信息
    task_storage[task_id] = {
        "task_id": task_id,
        "status": "processing",
        "query": request.query,
        "created_at": datetime.now(),
        "progress": 0.0,
        "message": "研究已启动"
    }
    
    # 在后台运行研究
    asyncio.create_task(
        _execute_research_task(
            task_id=task_id,
            query=request.query
        )
    )
    
    return ResearchResponse(
        task_id=task_id,
        status="processing",
        query=request.query,
        created_at=task_storage[task_id]["created_at"]
    )


@api_router.get("/research/history", response_model=List[ResearchResponse], tags=["research"])
async def get_research_history():
    """
    获取历史研究任务。
    这实现了我们架构中的事件流记录。
    """
    responses = []
    for task_id, task in task_storage.items():
        response = ResearchResponse(
            task_id=task_id,
            status=task["status"],
            query=task["query"],
            created_at=task["created_at"],
            completed_at=task.get("completed_at"),
            report=task.get("report"),
            sources=task.get("sources"),
            error=task.get("error")
        )
        responses.append(response)
    
    # 按创建时间排序，最新的在前
    responses.sort(key=lambda x: x.created_at, reverse=True)
    return responses


@api_router.get("/research/{task_id}", response_model=ResearchResponse, tags=["research"])
async def get_research_result(task_id: str):
    """
    获取研究任务的结果。
    """
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="任务未找到")
    
    task = task_storage[task_id]
    
    response = ResearchResponse(
        task_id=task_id,
        status=task["status"],
        query=task["query"],
        created_at=task["created_at"],
        completed_at=task.get("completed_at"),
        report=task.get("report"),
        sources=task.get("sources"),
        error=task.get("error")
    )
    
    return response


@api_router.get("/research/{task_id}/status", response_model=TaskStatusResponse, tags=["research"])
async def get_task_status(task_id: str):
    """
    获取研究任务的状态。
    """
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="任务未找到")
    
    task = task_storage[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", 0.0),
        message=task.get("message", ""),
        completed_at=task.get("completed_at")
    )


# 后台任务实现
async def _execute_research_task(
    task_id: str,
    query: str,
):
    """
    在后台执行研究任务。
    """
    try:
        task_storage[task_id]["progress"] = 0.1
        task_storage[task_id]["message"] = "初始化研究过程"
        
        from main import research_agent, memory_manager, report_generator, learning_manager
        from src.workflows.research_graph import ResearchGraphRunner
        runner = ResearchGraphRunner(research_agent, memory_manager, report_generator, __import__("config"), learning_manager)
        task_storage[task_id]["progress"] = 0.2
        task_storage[task_id]["message"] = "执行工作流"
        state = await runner.run(query, task_id=task_id)
        task_storage[task_id]["progress"] = 0.9
        task_storage[task_id]["message"] = "生成报告"
        report = state.get("report")
        research_result = state.get("research_result", {})
        task_storage[task_id]["progress"] = 1.0
        task_storage[task_id]["message"] = "研究成功完成"
        task_storage[task_id]["status"] = "completed"
        task_storage[task_id]["report"] = report
        task_storage[task_id]["sources"] = research_result.get("sources", [])
        task_storage[task_id]["metrics"] = state.get("evaluation_metrics", {})
        task_storage[task_id]["research_result"] = research_result
        task_storage[task_id]["completed_at"] = datetime.now()
        
        logger.info(f"研究任务 {task_id} 成功完成")
        
    except Exception as e:
        logger.error(f"研究任务 {task_id} 出错: {str(e)}")
        task_storage[task_id]["status"] = "failed"
        task_storage[task_id]["error"] = str(e)
        task_storage[task_id]["completed_at"] = datetime.now()
        task_storage[task_id]["progress"] = 1.0
        task_storage[task_id]["message"] = f"研究失败: {str(e)}"
@api_router.get("/research/{task_id}/download", tags=["research"])
async def download_report(task_id: str, format: str = "markdown"):
    from fastapi.responses import Response
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="任务未找到")
    task = task_storage[task_id]
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="任务未完成")
    content = task.get("report")
    if format in ("html", "markdown") and task.get("research_result"):
        from main import report_generator
        rr = task.get("research_result")
        content = await report_generator.generate_report(rr, format_type=format, include_sources=True)
    filename = f"report_{task_id}.{ 'html' if format=='html' else 'md'}"
    media = "text/html" if format == "html" else "text/markdown"
    return Response(content=content or "", media_type=media, headers={"Content-Disposition": f"attachment; filename={filename}"})
