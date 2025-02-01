"""
精准召回模块

该模块负责实现精准召回功能，通过分析查询内容，返回最匹配的记忆片段。
精准召回适用于用户需要查找具体信息或细节的场景。
"""
from typing import List, Dict
import logging
from models.memory_manager import MemoryManager

class PreciseRetriever:
    """
    精准召回类：负责从记忆库中精确检索与查询最相关的记忆片段。
    
    主要功能：
    1. 根据用户查询返回最相关的记忆片段
    2. 支持自定义召回数量和相似度阈值
    3. 对召回结果进行重排序，确保返回最相关的内容
    """
    
    def __init__(self, memory_manager: MemoryManager, 
                 default_top_k: int = 5,
                 min_similarity_score: float = 0.5):
        """
        初始化精准召回器
        
        参数:
            memory_manager: 记忆管理器实例
            default_top_k: 默认返回的记忆数量
            min_similarity_score: 最小相似度阈值，低于此阈值的记忆将被过滤
        """
        self.memory_manager = memory_manager
        self.default_top_k = default_top_k
        self.min_similarity_score = min_similarity_score
        logging.info("PreciseRetriever initialized with top_k=%d, min_score=%.2f", 
                    default_top_k, min_similarity_score)
    
    def retrieve_precise(self, query: str, user_id: str = "default_user", 
                        top_k: int = None) -> List[str]:
        """
        根据查询精准召回相关记忆片段
        
        参数:
            query: 用户查询文本
            user_id: 用户ID，默认为"default_user"
            top_k: 返回的记忆数量，如果为None则使用默认值
            
        返回:
            List[str]: 相关记忆片段列表，按相关度降序排列
        """
        try:
            # 使用实际的top_k值
            actual_top_k = top_k if top_k is not None else self.default_top_k
            
            # 调用记忆管理器检索记忆
            memories = self.memory_manager.retrieve_memories(
                user_id=user_id,
                query=query,
                top_k=actual_top_k
            )
            
            # 提取记忆内容并返回
            memory_contents = []
            for memory in memories:
                # 确保记忆对象包含必要的字段
                if isinstance(memory, dict) and 'content' in memory:
                    memory_contents.append(memory['content'])
                else:
                    logging.warning(f"Invalid memory format: {memory}")
            
            logging.info(f"Successfully retrieved {len(memory_contents)} memories for query: {query}")
            return memory_contents
            
        except Exception as e:
            logging.error(f"Error in precise retrieval for query '{query}': {str(e)}")
            return []
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """
        设置相似度阈值
        
        参数:
            threshold: 新的相似度阈值（0.0-1.0）
        """
        if 0.0 <= threshold <= 1.0:
            self.min_similarity_score = threshold
            logging.info(f"Similarity threshold updated to {threshold}")
        else:
            logging.error(f"Invalid similarity threshold: {threshold}. Must be between 0.0 and 1.0")
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
