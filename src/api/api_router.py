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
    max_sources: int = Field(default=10, description="要收集的最大来源数")
    report_format: str = Field(default="markdown", description="报告的输出格式")
    include_sources: bool = Field(default=True, description="是否包含来源引用")
    depth: str = Field(default="comprehensive", description="研究深度（basic, standard, comprehensive）")


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
            research_agent=research_agent,
            query=request.query,
            max_sources=request.max_sources,
            report_format=request.report_format,
            include_sources=request.include_sources,
            depth=request.depth
        )
    )
    
    return ResearchResponse(
        task_id=task_id,
        status="processing",
        query=request.query,
        created_at=task_storage[task_id]["created_at"]
    )


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


# 后台任务实现
async def _execute_research_task(
    task_id: str,
    research_agent: ResearchAgent,
    query: str,
    max_sources: int,
    report_format: str,
    include_sources: bool,
    depth: str
):
    """
    在后台执行研究任务。
    """
    try:
        task_storage[task_id]["progress"] = 0.1
        task_storage[task_id]["message"] = "初始化研究过程"
        
        # 使用研究代理执行研究
        task_storage[task_id]["progress"] = 0.2
        task_storage[task_id]["message"] = "收集信息"
        
        research_result = await research_agent.research(
            query=query,
            max_sources=max_sources,
            depth=depth
        )
        
        task_storage[task_id]["progress"] = 0.8
        task_storage[task_id]["message"] = "生成报告"
        
        # 生成报告
        report = await research_agent.generate_report(
            research_result=research_result,
            format_type=report_format,
            include_sources=include_sources
        )
        
        task_storage[task_id]["progress"] = 1.0
        task_storage[task_id]["message"] = "研究成功完成"
        task_storage[task_id]["status"] = "completed"
        task_storage[task_id]["report"] = report
        task_storage[task_id]["sources"] = research_result.get("sources", [])
        task_storage[task_id]["completed_at"] = datetime.now()
        
        logger.info(f"研究任务 {task_id} 成功完成")
        
    except Exception as e:
        logger.error(f"研究任务 {task_id} 出错: {str(e)}")
        task_storage[task_id]["status"] = "failed"
        task_storage[task_id]["error"] = str(e)
        task_storage[task_id]["completed_at"] = datetime.now()
        task_storage[task_id]["progress"] = 1.0
        task_storage[task_id]["message"] = f"研究失败: {str(e)}"