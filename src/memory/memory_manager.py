"""
Memory Manager for Deep Research Report Generation Agent System
Implements the memory layer based on MemGPT's layered memory architecture
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pickle
import os
from dataclasses import dataclass, asdict
from enum import Enum

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

from config import (
    CHROMA_PERSIST_DIR,
    MAIN_CONTEXT_SIZE,
    EXTERNAL_CONTEXT_SIZE,
    MEMORY_SUMMARY_THRESHOLD
)

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory in the system"""
    MAIN_CONTEXT = "main_context"      # Primary working memory (short-term)
    EXTERNAL_STORAGE = "external_storage"  # Long-term memory (external context)
    SCRATCHPAD = "scratchpad"          # Temporary working memory
    CONVERSATION_HISTORY = "conversation_history"  # Historical interactions


@dataclass
class MemoryEntry:
    """Represents a single memory entry"""
    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    memory_type: MemoryType
    embedding: Optional[List[float]] = None


class MemoryManager:
    """
    Implements the memory layer based on MemGPT's layered memory architecture
    Combines main context (working memory) with external storage (long-term memory)
    """
    
    def __init__(self):
        self.main_context: List[MemoryEntry] = []  # Primary working memory
        self.scratchpad: List[MemoryEntry] = []    # Temporary workspace
        self.conversation_history: List[MemoryEntry] = []  # Historical context
        
        # Initialize external memory storage using ChromaDB
        self.embeddings = OpenAIEmbeddings()
        self.external_memory = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="external_memory"
        )
        
        # Initialize system instructions in main context
        self._initialize_system_instructions()
        
        logger.info("Memory Manager initialized with layered memory architecture")
    
    def _initialize_system_instructions(self):
        """Initialize the main context with system instructions"""
        system_instructions = MemoryEntry(
            id="system_instructions",
            content="You are a Deep Research Report Generation Agent. Your role is to conduct comprehensive research on topics, gather information from multiple sources, analyze the data, and generate detailed reports with proper citations.",
            metadata={"type": "system_instruction", "priority": "high"},
            timestamp=datetime.now(),
            memory_type=MemoryType.MAIN_CONTEXT
        )
        self.main_context.append(system_instructions)
    
    async def add_to_main_context(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add content to the main context (working memory)"""
        if metadata is None:
            metadata = {}
        
        entry_id = f"main_{int(datetime.now().timestamp())}_{hash(content) % 10000}"
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata,
            timestamp=datetime.now(),
            memory_type=MemoryType.MAIN_CONTEXT
        )
        
        self.main_context.append(entry)
        
        # Check if we need to summarize or move older items to external storage
        await self._manage_main_context_size()
        
        logger.debug(f"Added to main context: {entry_id[:12]}")
        return entry_id
    
    async def add_to_external_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add content to external memory storage (long-term memory)"""
        if metadata is None:
            metadata = {}
        
        entry_id = f"external_{int(datetime.now().timestamp())}_{hash(content) % 10000}"
        
        # Add to ChromaDB
        doc = Document(
            page_content=content,
            metadata={**metadata, "id": entry_id, "timestamp": datetime.now().isoformat()}
        )
        
        # Add document to the collection
        added_ids = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self.external_memory.add_documents([doc])
        )
        
        logger.debug(f"Added to external memory: {entry_id[:12]}")
        return entry_id
    
    async def add_to_scratchpad(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add temporary content to scratchpad"""
        if metadata is None:
            metadata = {}
        
        entry_id = f"scratch_{int(datetime.now().timestamp())}_{hash(content) % 10000}"
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata,
            timestamp=datetime.now(),
            memory_type=MemoryType.SCRATCHPAD
        )
        
        self.scratchpad.append(entry)
        return entry_id
    
    async def search_external_memory(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search external memory for relevant information"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.external_memory.similarity_search(query, k=k)
            )
            
            # Format results
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "id": doc.metadata.get("id", "unknown")
                })
            
            logger.debug(f"Found {len(formatted_results)} results from external memory")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching external memory: {str(e)}")
            return []
    
    async def get_main_context_content(self) -> List[Dict[str, Any]]:
        """Get all content from main context"""
        return [asdict(entry) for entry in self.main_context]
    
    async def get_scratchpad_content(self) -> List[Dict[str, Any]]:
        """Get all content from scratchpad"""
        return [asdict(entry) for entry in self.scratchpad]
    
    async def clear_scratchpad(self):
        """Clear the scratchpad memory"""
        self.scratchpad = []
        logger.debug("Scratchpad cleared")
    
    async def _manage_main_context_size(self):
        """Manage the size of main context to prevent overflow"""
        # Calculate approximate token count (rough estimation: 1 token ~ 4 characters)
        total_chars = sum(len(entry.content) for entry in self.main_context)
        
        if total_chars > MAIN_CONTEXT_SIZE:
            logger.info(f"Main context size ({total_chars} chars) exceeds threshold ({MAIN_CONTEXT_SIZE}), managing...")
            
            # Move older entries to external memory (except system instructions)
            entries_to_move = []
            for entry in self.main_context:
                if entry.metadata.get("type") != "system_instruction":
                    entries_to_move.append(entry)
            
            # Keep the most recent entries in main context
            keep_count = max(2, len(self.main_context) // 2)  # Keep at least 2 entries
            entries_to_keep = self.main_context[:len(self.main_context) - len(entries_to_move)] + \
                             entries_to_move[-keep_count:]
            
            # Move older entries to external memory
            for entry in entries_to_move[:-keep_count]:
                await self.add_to_external_memory(entry.content, entry.metadata)
            
            # Update main context
            self.main_context = entries_to_keep
            logger.info(f"Moved {len(entries_to_move) - keep_count} entries to external memory")
    
    async def get_conversation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        recent_history = self.conversation_history[-limit:] if len(self.conversation_history) > limit else self.conversation_history
        return [asdict(entry) for entry in recent_history]
    
    async def add_to_conversation_history(self, role: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add an entry to conversation history"""
        if metadata is None:
            metadata = {}
        
        entry_id = f"conv_{int(datetime.now().timestamp())}_{hash(content) % 10000}"
        entry = MemoryEntry(
            id=entry_id,
            content=content,
            metadata={**metadata, "role": role},
            timestamp=datetime.now(),
            memory_type=MemoryType.CONVERSATION_HISTORY
        )
        
        self.conversation_history.append(entry)
        return entry_id
    
    async def reset_memory(self):
        """Reset all memory components (for testing or new sessions)"""
        self.main_context = []
        self.scratchpad = []
        self.conversation_history = []
        
        # Reinitialize system instructions
        self._initialize_system_instructions()
        
        # Clear external memory
        # Note: In a real implementation, you might want to keep some long-term knowledge
        # For now, we'll just recreate the collection
        self.external_memory = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="external_memory"
        )
        
        logger.info("Memory reset completed")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        main_context_chars = sum(len(entry.content) for entry in self.main_context)
        scratchpad_chars = sum(len(entry.content) for entry in self.scratchpad)
        conversation_count = len(self.conversation_history)
        
        # Get count from external memory
        external_count = 0
        try:
            external_count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.external_memory._collection.count()
            )
        except:
            # If count fails, we'll just set it to 0
            external_count = 0
        
        return {
            "main_context": {
                "entries": len(self.main_context),
                "total_chars": main_context_chars,
                "estimated_tokens": main_context_chars // 4
            },
            "scratchpad": {
                "entries": len(self.scratchpad),
                "total_chars": scratchpad_chars,
                "estimated_tokens": scratchpad_chars // 4
            },
            "conversation_history": {
                "entries": conversation_count
            },
            "external_memory": {
                "entries": external_count
            },
            "timestamp": datetime.now().isoformat()
        }