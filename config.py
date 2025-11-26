"""
Configuration file for Deep Research Report Generation Agent System
Based on the system architecture design document
"""
import os
from typing import Optional

# LLM Configuration
LLM_BASE_MODEL = os.getenv("LLM_BASE_MODEL", "gpt-4-turbo")  # Based on architecture recommendation
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# Vector Database Configuration
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma")  # Based on architecture recommendation
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Memory Layer Configuration (Based on MemGPT's layered memory architecture)
MAIN_CONTEXT_SIZE = int(os.getenv("MAIN_CONTEXT_SIZE", "8192"))  # Tokens in main context
EXTERNAL_CONTEXT_SIZE = int(os.getenv("EXTERNAL_CONTEXT_SIZE", "1000000"))  # Max external context
MEMORY_SUMMARY_THRESHOLD = int(os.getenv("MEMORY_SUMMARY_THRESHOLD", "1000"))  # When to summarize

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() == "true"

# Agent Configuration
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "300"))  # 5 minutes timeout
MAX_TOOL_CALLS_PER_STEP = int(os.getenv("MAX_TOOL_CALLS_PER_STEP", "10"))

# Tool Configuration
BROWSER_TOOL_ENABLED = os.getenv("BROWSER_TOOL_ENABLED", "True").lower() == "true"
CODE_EXECUTION_ENABLED = os.getenv("CODE_EXECUTION_ENABLED", "True").lower() == "true"
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))

# Report Generation Configuration
REPORT_TEMPLATE_PATH = os.getenv("REPORT_TEMPLATE_PATH", "./templates")
DEFAULT_REPORT_FORMAT = os.getenv("DEFAULT_REPORT_FORMAT", "markdown")

# Security and Performance
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "./logs/app.log")