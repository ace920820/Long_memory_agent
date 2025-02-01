"""
记忆召回管理模块

该模块作为记忆召回的统一入口，根据查询意图选择合适的召回策略。
支持精准召回（返回具体记忆片段）和综合召回（返回记忆簇摘要）两种方式。
"""
from typing import Dict, Optional
import logging
from models.query_classifier import QueryClassifier
from models.precise_retriever import PreciseRetriever
from models.comprehensive_retriever import ComprehensiveRetriever
from models.memory_manager import MemoryManager


class RetrievalManager:
    """
    记忆召回管理类：作为记忆召回的统一入口，协调不同的召回策略。
    
    主要功能：
    1. 解析查询意图，选择合适的召回策略
    2. 统一调用精准召回和综合召回
    3. 处理上下文信息，优化召回结果
    """
    
    def __init__(self, memory_manager: MemoryManager):
        """
        初始化召回管理器
        
        参数:
            memory_manager: 记忆管理器实例
        """
        self.memory_manager = memory_manager
        
        # 初始化查询分类器和召回器
        self.query_classifier = QueryClassifier()
        self.precise_retriever = PreciseRetriever(memory_manager)
        self.comprehensive_retriever = ComprehensiveRetriever(memory_manager)
        
        logging.info("RetrievalManager initialized with all components")
    
    def retrieve_memory(self, query: str, context: Dict = None, 
                       user_id: str = "default_user") -> Optional[str]:
        """
        根据查询和上下文检索记忆
        
        参数:
            query: 用户查询文本
            context: 上下文信息字典，可选
            user_id: 用户ID，默认为"default_user"
            
        返回:
            str: 召回的记忆内容（记忆片段或记忆簇摘要），如果没有找到相关记忆则返回None
        """
        try:
            if not query:
                logging.warning("Empty query received")
                return None
                
            # 使用查询分类器确定查询意图
            retrieval_type = self.query_classifier.classify_query(query)
            logging.info(f"Query '{query}' classified as {retrieval_type}")
            
            # 根据查询意图选择召回策略
            if retrieval_type == "COMPREHENSIVE":
                # 使用综合召回，返回记忆簇摘要
                result = self.comprehensive_retriever.retrieve_comprehensive(
                    query=query,
                    user_id=user_id
                )
                if result:
                    logging.info("Successfully retrieved comprehensive memory summary")
                else:
                    logging.info("No comprehensive memory found")
                    
            else:  # PRECISE
                # 使用精准召回，返回具体记忆片段
                memories = self.precise_retriever.retrieve_precise(
                    query=query,
                    user_id=user_id
                )
                if memories:
                    # 如果有多个记忆片段，将它们组合成一个结果
                    result = "\n".join(memories)
                    logging.info(f"Successfully retrieved {len(memories)} precise memories")
                else:
                    result = None
                    logging.info("No precise memories found")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in memory retrieval for query '{query}': {str(e)}")
            return None
    
    def update_retrieval_config(self, precise_top_k: int = None,
                              precise_threshold: float = None,
                              comprehensive_threshold: float = None) -> None:
        """
        更新召回配置参数
        
        参数:
            precise_top_k: 精准召回返回的记忆数量
            precise_threshold: 精准召回的相似度阈值
            comprehensive_threshold: 综合召回的相似度阈值
        """
        try:
            if precise_top_k is not None:
                self.precise_retriever.default_top_k = precise_top_k
                logging.info(f"Updated precise retrieval top_k to {precise_top_k}")
                
            if precise_threshold is not None:
                self.precise_retriever.set_similarity_threshold(precise_threshold)
                logging.info(f"Updated precise retrieval threshold to {precise_threshold}")
                
            if comprehensive_threshold is not None:
                self.comprehensive_retriever.set_similarity_threshold(comprehensive_threshold)
                logging.info(f"Updated comprehensive retrieval threshold to {comprehensive_threshold}")
                
        except ValueError as e:
            logging.error(f"Error updating retrieval config: {str(e)}")
            raise
