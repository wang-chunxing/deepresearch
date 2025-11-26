"""
API router for Deep Research Report Generation Agent System
Based on the interaction layer design from architecture document
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

# Request/Response models
class ResearchRequest(BaseModel):
    query: str = Field(..., description="The research query or topic to investigate")
    max_sources: int = Field(default=10, description="Maximum number of sources to gather")
    report_format: str = Field(default="markdown", description="Output format for the report")
    include_sources: bool = Field(default=True, description="Whether to include source citations")
    depth: str = Field(default="comprehensive", description="Depth of research (basic, standard, comprehensive)")


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


# In-memory task storage (in production, use a database)
task_storage: Dict[str, Dict[str, Any]] = {}


@api_router.post("/research", response_model=ResearchResponse, tags=["research"])
async def start_research(request: ResearchRequest):
    """
    Start a new research task based on the provided query.
    This implements the interaction layer from our architecture.
    """
    from main import research_agent  # Import here to avoid circular import
    
    task_id = f"task_{int(datetime.now().timestamp())}_{hash(request.query) % 10000}"
    
    # Store initial task info
    task_storage[task_id] = {
        "task_id": task_id,
        "status": "processing",
        "query": request.query,
        "created_at": datetime.now(),
        "progress": 0.0,
        "message": "Research initiated"
    }
    
    # Run research in background
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
    Get the result of a research task.
    """
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
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
    Get the status of a research task.
    """
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
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
    Get historical research tasks.
    This implements the event flow recording from our architecture.
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
    
    # Sort by creation time, most recent first
    responses.sort(key=lambda x: x.created_at, reverse=True)
    return responses


# Background task implementation
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
    Execute the research task in the background.
    """
    try:
        task_storage[task_id]["progress"] = 0.1
        task_storage[task_id]["message"] = "Initializing research process"
        
        # Perform the research using the research agent
        task_storage[task_id]["progress"] = 0.2
        task_storage[task_id]["message"] = "Gathering information"
        
        research_result = await research_agent.research(
            query=query,
            max_sources=max_sources,
            depth=depth
        )
        
        task_storage[task_id]["progress"] = 0.8
        task_storage[task_id]["message"] = "Generating report"
        
        # Generate the report
        report = await research_agent.generate_report(
            research_result=research_result,
            format_type=report_format,
            include_sources=include_sources
        )
        
        task_storage[task_id]["progress"] = 1.0
        task_storage[task_id]["message"] = "Research completed successfully"
        task_storage[task_id]["status"] = "completed"
        task_storage[task_id]["report"] = report
        task_storage[task_id]["sources"] = research_result.get("sources", [])
        task_storage[task_id]["completed_at"] = datetime.now()
        
        logger.info(f"Research task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in research task {task_id}: {str(e)}")
        task_storage[task_id]["status"] = "failed"
        task_storage[task_id]["error"] = str(e)
        task_storage[task_id]["completed_at"] = datetime.now()
        task_storage[task_id]["progress"] = 1.0
        task_storage[task_id]["message"] = f"Research failed: {str(e)}"