"""
Main application entry point for Deep Research Report Generation Agent System
Based on the system architecture design document
"""
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import configuration
from config import (
    API_HOST, 
    API_PORT, 
    DEBUG_MODE, 
    LOG_LEVEL,
    LOG_FILE
)

# Import core modules based on architecture
from src.api.api_router import api_router
from src.agents.research_agent import ResearchAgent
from src.memory.memory_manager import MemoryManager
from src.generation.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Deep Research Report Generation Agent System",
    description="An advanced AI system for generating comprehensive research reports using multi-agent collaboration and long-term memory management",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(api_router, prefix="/api/v1", tags=["research-agent"])

# Mount frontend static files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# Initialize core components
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Deep Research Report Generation Agent System...")
    
    # Initialize memory manager
    global memory_manager
    memory_manager = MemoryManager()
    
    # Initialize report generator
    global report_generator
    report_generator = ReportGenerator()
    
    # Initialize research agent
    global research_agent
    research_agent = ResearchAgent(
        memory_manager=memory_manager,
        report_generator=report_generator
    )
    
    logger.info("System components initialized successfully")

@app.get("/")
async def root():
    return {
        "message": "Deep Research Report Generation Agent System",
        "status": "running",
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "api": "operational",
            "memory": "operational",  # This would check actual memory status in real implementation
            "llm": "operational"  # This would check actual LLM connectivity in real implementation
        }
    }

if __name__ == "__main__":
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=8001,  # 直接使用8001端口
        reload=DEBUG_MODE,
        log_level=LOG_LEVEL.lower()
    )