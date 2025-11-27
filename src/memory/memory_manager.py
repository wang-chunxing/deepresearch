"""
内存管理器用于深度研究报告生成代理系统
基于MemGPT的分层内存架构实现内存层
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
from langchain_core.documents import Document

from config import (
    CHROMA_PERSIST_DIR,
    MAIN_CONTEXT_SIZE,
    EXTERNAL_CONTEXT_SIZE,
    MEMORY_SUMMARY_THRESHOLD
)

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """系统中的内存类型"""
    MAIN_CONTEXT = "main_context"      # 主工作内存（短期）
    EXTERNAL_STORAGE = "external_storage"  # 长期内存（外部上下文）
    SCRATCHPAD = "scratchpad"          # 临时工作内存
    CONVERSATION_HISTORY = "conversation_history"  # 历史交互


@dataclass
class MemoryEntry:
    """表示单个内存条目"""
    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    memory_type: MemoryType
    embedding: Optional[List[float]] = None


class MemoryManager:
    """
    基于MemGPT的分层内存架构实现内存层
    结合主上下文（工作内存）和外部存储（长期内存）
    """
    
    def __init__(self):
        self.main_context: List[MemoryEntry] = []  # 主工作内存
        self.scratchpad: List[MemoryEntry] = []    # 临时工作区
        self.conversation_history: List[MemoryEntry] = []  # 历史交互
        
        # 使用ChromaDB初始化外部内存存储
        self.embeddings = OpenAIEmbeddings()
        self.external_memory = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="external_memory"
        )
        
        # 在主上下文中初始化系统指令
        self._initialize_system_instructions()
        
        logger.info("内存管理器已使用分层内存架构初始化")
    
    def _initialize_system_instructions(self):
        """初始化主上下文的系统指令"""
        system_instructions = MemoryEntry(
            id="system_instructions",
            content="您是一个深度研究报告生成代理。您的角色是就主题进行全面研究，从多个来源收集信息，分析数据，并生成带有适当引用的详细报告。",
            metadata={"type": "system_instruction", "priority": "high"},
            timestamp=datetime.now(),
            memory_type=MemoryType.MAIN_CONTEXT
        )
        self.main_context.append(system_instructions)
    
    async def add_to_main_context(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """向主上下文（工作内存）添加内容"""
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
        
        # 检查是否需要总结或移动较旧的项目到外部存储
        await self._manage_main_context_size()
        
        logger.debug(f"添加到主上下文: {entry_id[:12]}")
        return entry_id
    
    async def add_to_external_memory(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """向外部内存存储（长期内存）添加内容"""
        if metadata is None:
            metadata = {}
        
        entry_id = f"external_{int(datetime.now().timestamp())}_{hash(content) % 10000}"
        
        # 添加到ChromaDB
        doc = Document(
            page_content=content,
            metadata={**metadata, "id": entry_id, "timestamp": datetime.now().isoformat()}
        )
        
        # 添加文档到集合
        added_ids = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: self.external_memory.add_documents([doc])
        )
        
        logger.debug(f"添加到外部内存: {entry_id[:12]}")
        return entry_id
    
    async def add_to_scratchpad(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """向临时工作区添加临时内容"""
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
        """搜索外部内存的相关信息"""
        try:
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.external_memory.similarity_search(query, k=k)
            )
            
            # 格式化结果
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "id": doc.metadata.get("id", "unknown")
                })
            
            logger.debug(f"从外部内存找到 {len(formatted_results)} 个结果")
            return formatted_results
        except Exception as e:
            logger.error(f"搜索外部内存时出错: {str(e)}")
            return []
    
    async def get_main_context_content(self) -> List[Dict[str, Any]]:
        """获取主上下文的所有内容"""
        return [asdict(entry) for entry in self.main_context]
    
    async def get_scratchpad_content(self) -> List[Dict[str, Any]]:
        """获取临时工作区的所有内容"""
        return [asdict(entry) for entry in self.scratchpad]
    
    async def clear_scratchpad(self):
        """清除临时工作区内存"""
        self.scratchpad = []
        logger.debug("临时工作区已清除")
    
    async def _manage_main_context_size(self):
        """管理主上下文大小以防止溢出"""
        # 计算近似令牌数（粗略估计：1令牌 ~ 4个字符）
        total_chars = sum(len(entry.content) for entry in self.main_context)
        
        if total_chars > MAIN_CONTEXT_SIZE:
            logger.info(f"主上下文大小 ({total_chars} 个字符) 超过阈值 ({MAIN_CONTEXT_SIZE})，正在管理...")
            
            # 将较旧的条目移动到外部内存（系统指令除外）
            entries_to_move = []
            for entry in self.main_context:
                if entry.metadata.get("type") != "system_instruction":
                    entries_to_move.append(entry)
            
            # 在主上下文中保留最新条目
            keep_count = max(2, len(self.main_context) // 2)  # 至少保留2个条目
            entries_to_keep = self.main_context[:len(self.main_context) - len(entries_to_move)] + \
                             entries_to_move[-keep_count:]
            
            # 将较旧的条目移动到外部内存
            for entry in entries_to_move[:-keep_count]:
                await self.add_to_external_memory(entry.content, entry.metadata)
            
            # 更新主上下文
            self.main_context = entries_to_keep
            logger.info(f"已将 {len(entries_to_move) - keep_count} 个条目移动到外部内存")
    
    async def get_conversation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近的对话历史"""
        recent_history = self.conversation_history[-limit:] if len(self.conversation_history) > limit else self.conversation_history
        return [asdict(entry) for entry in recent_history]
    
    async def add_to_conversation_history(self, role: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """向对话历史添加条目"""
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
        """重置所有内存组件（用于测试或新会话）"""
        self.main_context = []
        self.scratchpad = []
        self.conversation_history = []
        
        # 重新初始化系统指令
        self._initialize_system_instructions()
        
        # 清除外部内存
        # 注意：在实际实现中，您可能希望保留一些长期知识
        # 目前，我们只是重新创建集合
        self.external_memory = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="external_memory"
        )
        
        logger.info("内存重置完成")
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存使用统计信息"""
        main_context_chars = sum(len(entry.content) for entry in self.main_context)
        scratchpad_chars = sum(len(entry.content) for entry in self.scratchpad)
        conversation_count = len(self.conversation_history)
        
        # 从外部内存获取计数
        external_count = 0
        try:
            external_count = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.external_memory._collection.count()
            )
        except:
            # 如果计数失败，我们将其设置为0
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